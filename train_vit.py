import torch
import os
import tempfile
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.v8_api_enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # 
torch.backends.cuda.allow_tensor_float_32 = True

# Make Triton and TorchInductor use stable, user-writable cache/tmp dirs on the cluster
os.environ.setdefault("TRITON_CACHE_DIR", "/home/ethan/.triton/cache")
os.environ.setdefault("TMPDIR", "/home/ethan/tmp")
tempfile.tempdir = os.environ["TMPDIR"]

import json
import os
from pathlib import Path
from typing import Tuple
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from vit import VisionTransformer
from randomaug import RandAugment


def build_transforms(
    image_size: int, do_augs=True,
) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    train_tfms_list = [
        transforms.RandomCrop(image_size, padding=4),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
    ]
    if do_augs:
        train_tfms_list.insert(0, RandAugment(n=2, m=14))

    train_tfms_list.extend([transforms.ToTensor(), normalize])

    train_tfms = transforms.Compose(train_tfms_list)
    val_tfms = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])
    return train_tfms, val_tfms


def get_dataloaders(
    data_dir: str, image_size: int, batch_size: int, num_workers: int, val_split: int, do_augs: bool
):
    train_tfms, val_tfms = build_transforms(image_size, do_augs=do_augs)
    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    val_len = val_split
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(
        full_train, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset.transform = val_tfms  # type: ignore[attr-defined]

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def build_param_groups(model: torch.nn.Module, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def evaluate(model, dataloader, accelerator):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(accelerator.device), labels.to(accelerator.device)
            loss, logits = model(images, labels)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def main():
    args = {
        "data_dir": "./data",
        "output_dir": "./vit-cifar-out",
        "epochs": 30,
        "batch_size": 256,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "num_workers": 12,
        "image_size": 32,
        "patch_size": 4,
        "do_augs": True,
        "dim": 256,
        "depth": 7,
        "heads": 4,
        "ff_mult": 4,
        "ff_mode": "vanilla",  # vanilla | hyper | alternate
        "use_cls_token": True,
        "gradient_checkpointing": False,
        "mixed_precision": "bf16",  # no | fp16 | bf16
        "compile": False,
        "seed": 123,
        "val_split": 5000,
        "with_tracking": True,
        "report_to": "wandb",
        "wandb_project": "vit-cifar-hyp",
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    args = SimpleNamespace(**args)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.with_tracking else None,
        project_dir=args.output_dir,
    )
    set_seed(args.seed, device_specific=True)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print("Args:", json.dumps(vars(args), indent=2))

    if args.with_tracking:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.ff_mode}},
        )

    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        args.image_size,
        args.batch_size,
        args.num_workers,
        args.val_split,
        args.do_augs,
    )

    model = VisionTransformer(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        ff_mult=args.ff_mult,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=10,
        in_channels=3,
        use_cls_token=args.use_cls_token,
        gradient_checkpointing=args.gradient_checkpointing,
        ff_mode=args.ff_mode,
    )

    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.weight_decay),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.compile:
        model = torch.compile(model, mode="default", fullgraph=True)

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    best_val_acc = 0.0
    progress = tqdm(range(args.epochs), disable=not accelerator.is_local_main_process)
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(accelerator.device), labels.to(accelerator.device)
            with accelerator.accumulate(model):
                loss, logits = model(images, labels)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            accelerator.log({"train/loss": loss.item()}, step=epoch)

        lr_scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, accelerator)
        accelerator.log({"val/loss": val_loss, "val/acc": val_acc}, step=epoch)
        best_val_acc = max(best_val_acc, val_acc)

        if accelerator.is_local_main_process:
            tqdm.write(f"Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        # if val_acc > best_val_acc and accelerator.is_main_process:
        #     best_val_acc = val_acc
        #     unwrapped = accelerator.unwrap_model(model)
        #     save_path = Path(args.output_dir) / "best.pt"
        #     torch.save(unwrapped.state_dict(), save_path)

        progress.update(1)

    if accelerator.is_main_process:
        with open(Path(args.output_dir) / "metrics.json", "w") as f:
            json.dump({"val_acc": best_val_acc}, f, indent=2)

    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
