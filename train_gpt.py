import os
import tempfile
import inspect as _inspect
from pathlib import Path

# Ensure Hugging Face datasets cache is set early to a user-writable path.
os.environ.setdefault("HF_HOME", "/fsx/scratch/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))

# Make Triton and TorchInductor use stable, user-writable cache/tmp dirs on the cluster
os.environ.setdefault("TRITON_CACHE_DIR", "/home/ethan/.triton/cache")
os.environ.setdefault("TMPDIR", "/home/ethan/tmp")
tempfile.tempdir = os.environ["TMPDIR"]

# When using datasets.map(num_proc=...) we rely on multiprocessing for parallelism,
# so we disable internal tokenizer threading to avoid oversubscription.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import argparse
import json
import logging
import math
import os
from itertools import chain
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)
from types import SimpleNamespace

import torch
import torch.nn as nn
from typing import Optional
import time

from param_tracker import ParameterTracker
from transformer import Transformer


logger = get_logger(__name__)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.v8_api_enabled = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True # 
torch.backends.cuda.allow_tensor_float_32 = True

import torch._dynamo as dynamo
dynamo.config.verbose = False
dynamo.config.suppress_errors = True
dynamo.config.assume_static_by_default = True  # if shapes rarely change
dynamo.config.cache_size_limit = 64  # limit graph cache to avoid recompiles
dynamo.config.guard_nn_modules = True  # avoids excess guards

import torch._inductor as inductor
inductor.config.triton.cudagraphs = True   # or False if capture causes overhead
inductor.config.max_autotune = False       # skip exhaustive autotune when compile time matters
inductor.config.use_mixed_mm = True        # enables faster matmul codegen


# torch.set_num_threads(n)
# torch.set_num_interop_threads(m)

def main():

    parser = argparse.ArgumentParser(
        description="Train GPT with optional JSON overrides for sweeps.",
    )
    parser.add_argument(
        "--override_json",
        type=str,
        default=None,
        help="Optional path to a JSON file whose keys override the defaults below.",
    )
    cli_args = parser.parse_args()

    args = {
        "num_validation_batches": 25,
        "validate_every": 1000,
        # "dataset_name": "wikitext",
        # "dataset_config_name": "wikitext-103-v1",
        "dataset_name": "openwebtext",
        "dataset_config_name": None,
        "train_file": None,
        "validation_file": None,
        "validation_split_percentage": 5,
        "model_name_or_path": "openai-community/gpt2-medium",
        # "model_name_or_path": "openai-community/gpt2",
        "config_name": None,
        "tokenizer_name": None,
        "use_slow_tokenizer": False,
        "per_device_train_batch_size": 32,
        "learning_rate": 3.0e-4,
        "weight_decay": 0.01,
        "num_train_epochs": 2,
        "max_train_steps": 200_000,
        "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 250,
        "seed": 123,
        "model_type": None,
        "block_size": 1024,
        "preprocessing_num_workers": 180,
        "overwrite_cache": False,
        "no_keep_linebreaks": False,
        "trust_remote_code": False,
        "checkpointing_steps": None,
        "resume_from_checkpoint": None,
        "with_tracking": True,
        "report_to": "wandb",
        "low_cpu_mem_usage": False,
        "max_grad_norm": 1.0,
        "hf_path": None,
        "base_output_dir": "model-output",

        "hidden_size": 1024,
        "depth": 12,
        "n_head": 8,

        # Architecture options
        "use_rope": True,           # Use rotary position embeddings (else learned)
        "rope_theta": 10000.0,      # RoPE frequency base
        "qk_norm": True,            # Apply QK normalization before attention

        "beta1": 0.9,
        "beta2": 0.999,

        "compile": True,
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": True,

        "gradient_checkpointing": True,

        "num_workers": 12,

        # FFN type selection: mlp, swiglu, lowrank_hypergate, lowrank_hyperffn, spectral, butterfly, composite
        "ffn_type": "swiglu",
        "kron_p": 64,              # Kronecker factor size p (dim = p * q)
        "kron_q": 64,              # Kronecker factor size q
        "lowrank_rank": 16,        # Rank for LowRankHyperGate/LowRankHyperFFN
        # LowRank delta target (for lowrank_hypergate - SwiGLU): 
        #   input-space: gate_input, linear_input, both_input
        #   output-space: gate_output, linear_output, both_output, w2_output, full
        "lowrank_delta_target": "gate_input",
        # FFN delta target (for lowrank_hyperffn - plain FFN):
        #   w1_input, w1_output, w2_input, w2_output, full
        "ffn_delta_target": "w1_output",
        # LowRank factor type: lowrank (U@V^T), kronecker (A⊗B)
        "lowrank_factor_type": "lowrank",
        "lowrank_use_bmm": True,   # True for bmm/matmul (better TF32), False for einsum
        "butterfly_rounds": 3,     # Rounds for ButterflyTransformLayer
        "spectral_bottleneck": None,  # Bottleneck dim for SpectralModulation mask gen (None = direct)
        # Spectral transform type: "fft" (complex), "dct" (real), "hadamard" (real, power-of-2)
        "spectral_transform": "dct",
        # Spectral mask activation: "tanh", "silu", "none", "softsign"
        "spectral_mask_act": "none",
        # Spectral additive branch: inject token-dependent frequency content
        "spectral_use_additive": False,
        # Base weight mode for LowRankHyperFFN:
        #   "regular": standard nn.Linear weights
        #   "lowrank_N": base weights as low-rank matrices of rank N (e.g., "lowrank_8")
        #   "none": no base weights (pure dynamic)
        "base_weight_mode": "regular",
        # Third layer: optional additional residual block after FFN
        # Options: None (disabled), "spectral", "butterfly", "lowrank_hypergate", "lowrank_hyperffn"
        "third_layer_type": None,

        "log_params_every_n": 100,
        "activation_sample_limit": 2,

        "hf_cache_dir": "/fsx/scratch/huggingface/hub",

    }

    if cli_args.override_json is not None:
        with open(cli_args.override_json, "r") as f:
            json_overrides = json.load(f)
        if not isinstance(json_overrides, dict):
            raise ValueError("Override JSON must contain a top-level object/dictionary.")
        args.update(json_overrides)

    config = AutoConfig.from_pretrained(
        args['model_name_or_path'],
        trust_remote_code=args['trust_remote_code'],
    )
    vocab_size = config.vocab_size
    # Build a descriptive run/output name from key experimental settings
    ffn_type = args['ffn_type']
    run_name = ffn_type
    if ffn_type == 'composite':
        run_name += f"_p{args['kron_p']}q{args['kron_q']}"
    elif ffn_type == 'lowrank_hypergate':
        factor_type = args.get('lowrank_factor_type', 'lowrank')
        if factor_type == 'kronecker':
            run_name += "_kron"
        else:
            run_name += f"_r{args['lowrank_rank']}"
        run_name += f"_{args['lowrank_delta_target']}"
        if not args.get('lowrank_use_bmm', True):
            run_name += "_einsum"
    elif ffn_type == 'lowrank_hyperffn':
        factor_type = args.get('lowrank_factor_type', 'lowrank')
        if factor_type == 'kronecker':
            run_name += "_kron"
        else:
            run_name += f"_r{args['lowrank_rank']}"
        run_name += f"_{args['ffn_delta_target']}"
        base_mode = args.get('base_weight_mode', 'regular')
        if base_mode != 'regular':
            run_name += f"_{base_mode}"
        if not args.get('lowrank_use_bmm', True):
            run_name += "_einsum"
    elif ffn_type == 'butterfly':
        run_name += f"_r{args['butterfly_rounds']}"
    elif ffn_type == 'spectral':
        transform = args.get('spectral_transform', 'dct')
        run_name += f"_{transform}"
        mask_act = args.get('spectral_mask_act', 'none')
        if mask_act != 'none':
            run_name += f"_{mask_act}"
        if args.get('spectral_use_additive', False):
            run_name += "_add"
        bn = args.get('spectral_bottleneck')
        if bn is not None:
            run_name += f"_bn{bn}"

    # Add third layer type to run name if set
    third_layer = args.get('third_layer_type')
    if third_layer is not None:
        run_name += f"+{third_layer}"
        # Add relevant params for the third layer
        if third_layer in ('lowrank_hypergate', 'lowrank_hyperffn'):
            factor_type = args.get('lowrank_factor_type', 'lowrank')
            if factor_type == 'kronecker':
                run_name += "_kron"
            else:
                run_name += f"_r{args['lowrank_rank']}"
        elif third_layer == 'butterfly':
            run_name += f"_r{args['butterfly_rounds']}"
        elif third_layer == 'spectral':
            transform = args.get('spectral_transform', 'dct')
            run_name += f"_{transform}"
            mask_act = args.get('spectral_mask_act', 'none')
            if mask_act != 'none':
                run_name += f"_{mask_act}"
            if args.get('spectral_use_additive', False):
                run_name += "_add"
            bn = args.get('spectral_bottleneck')
            if bn is not None:
                run_name += f"_bn{bn}"

    args["run_name"] = run_name
    args["output_dir"] = f"{args['base_output_dir']}/{run_name}"

    args = SimpleNamespace(**args)

    print("Running with the following arguments:")
    print(json.dumps(vars(args), indent=2))

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.output_dir is None:
        args.output_dir = time.strftime("run_%Y%m%d_%H%M%S")

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                                                            mixed_precision="bf16",
                                                            **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split={
            "train": f"train[{args.validation_split_percentage}%:]",
            "validation": f"train[:{args.validation_split_percentage}%]",
        },
        cache_dir=args.hf_cache_dir,
        num_proc=args.preprocessing_num_workers,
        # download_mode="force_redownload",
        # download_config=DownloadConfig(num_proc=args.preprocessing_num_workers),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )


    model = Transformer(
        dim=args.hidden_size,
        depth=args.depth,
        heads=args.n_head,
        ff_mult=4,
        vocab_size=vocab_size,
        max_seq_len=args.block_size,
        gradient_checkpointing=args.gradient_checkpointing,
        use_rope=getattr(args, 'use_rope', True),
        rope_theta=getattr(args, 'rope_theta', 10000.0),
        qk_norm=getattr(args, 'qk_norm', True),
        ffn_type=getattr(args, 'ffn_type', 'swiglu'),
        kron_p=getattr(args, 'kron_p', 64),
        kron_q=getattr(args, 'kron_q', 64),
        lowrank_rank=getattr(args, 'lowrank_rank', 16),
        lowrank_delta_target=getattr(args, 'lowrank_delta_target', 'gate_input'),
        ffn_delta_target=getattr(args, 'ffn_delta_target', 'w1_output'),
        lowrank_factor_type=getattr(args, 'lowrank_factor_type', 'lowrank'),
        lowrank_use_bmm=getattr(args, 'lowrank_use_bmm', True),
        butterfly_rounds=getattr(args, 'butterfly_rounds', 3),
        spectral_bottleneck=getattr(args, 'spectral_bottleneck', None),
        spectral_transform=getattr(args, 'spectral_transform', 'dct'),
        spectral_mask_act=getattr(args, 'spectral_mask_act', 'none'),
        spectral_use_additive=getattr(args, 'spectral_use_additive', False),
        base_weight_mode=getattr(args, 'base_weight_mode', 'regular'),
        third_layer_type=getattr(args, 'third_layer_type', None),
    )

    print(model)

    print("num parameters", sum(p.numel() for p in model.parameters()))
    model = model.to(accelerator.device)
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.token_embedding.weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("resizing token embeddings", len(tokenizer), embedding_size)
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets: tokenize and group in one pass to avoid
    # caching an intermediate tokenized dataset.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    def tokenize_and_group(examples):
        tokenized = tokenize_function(examples)
        concatenated_examples = {k: list(chain(*tokenized[k])) for k in tokenized.keys()}
        total_length = len(concatenated_examples[list(tokenized.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts would drop a remainder
    # per batch. The fused tokenizer+grouper keeps the same behavior while saving one dataset cache.
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            tokenize_and_group,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Tokenize + group into {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers, pin_memory=True
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_mult = getattr(param, "lr_mult", 1.0)
        optimizer_grouped_parameters.append(
            {
                "params": [param],
                "weight_decay": 0.0 if any(nd in name for nd in no_decay) else args.weight_decay,
                "lr": args.learning_rate * lr_mult,
            }
        )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1, args.beta2), fused=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # compile
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=args.compile_fullgraph)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]
        wandb_name = getattr(args, "run_name", None)
        init_kwargs = {
            "wandb":
                {
                    "name": wandb_name,
                }
        }
        accelerator.init_trackers("BabyHypernetworks", experiment_config, init_kwargs=init_kwargs)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)


    # allocated and reserved memory
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    progress_bar.set_postfix(vram=f"{reserved_memory / (1024 ** 3):.2f} GB")

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    data_start = torch.cuda.Event(enable_timing=True)
    data_end = torch.cuda.Event(enable_timing=True)
    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        dataloader_iter = iter(active_dataloader)
        for step in range(len(active_dataloader)):
            model.train()
            
            # Time data loading
            data_start.record()
            batch = next(dataloader_iter)
            data_end.record()
            
            with accelerator.accumulate(model):
                # Time forward pass
                forward_start.record()
                # logits, loss = model(idx=batch["input_ids"], targets=batch["labels"])
                input_ids = batch["input_ids"].to(accelerator.device)[:, :-1]
                targets = batch["labels"].to(accelerator.device)[:, 1:]
                loss, logits = model(input_ids=input_ids, targets=targets)
                forward_end.record()
                
                # Sync loss across GPUs for accurate metrics (needed for multi-GPU training)
                synced_loss = accelerator.gather(loss.detach().float()).mean()
                
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += synced_loss
                
                # Time backward pass
                backward_start.record()
                accelerator.backward(loss)
                backward_end.record()
                
                # Synchronize to get accurate timings
                torch.cuda.synchronize()
                
                # Calculate elapsed times in milliseconds
                data_time = data_start.elapsed_time(data_end)
                forward_time = forward_start.elapsed_time(forward_end)
                backward_time = backward_start.elapsed_time(backward_end)
                
                # clip the gradients
                mini_logs ={
                        "step_loss": synced_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "timer/data_load_ms": data_time,
                        "timer/forward_ms": forward_time,
                        "timer/backward_ms": backward_time,
                    }

                if args.max_grad_norm is not None:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    mini_logs["grad_norm"] = grad_norm
                
                # Time optimizer step
                optimizer_start.record()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_end.record()
                torch.cuda.synchronize()
                
                optimizer_time = optimizer_start.elapsed_time(optimizer_end)
                mini_logs["timer/optimizer_ms"] = optimizer_time
                
                accelerator.log(
                        mini_logs,
                        step=completed_steps,
                    )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

            
            if completed_steps % args.validate_every == 0:
                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        input_ids = batch["input_ids"].to(accelerator.device)[:, :-1]
                        targets = batch["labels"].to(accelerator.device)[:, 1:]
                        loss, logits = model(input_ids=input_ids, targets=targets)

                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_train_batch_size)))
                    if args.num_validation_batches is not None:
                        if step >= args.num_validation_batches:
                            break

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

                if args.with_tracking:
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "train_loss": total_loss.item() / len(train_dataloader),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        print("Saving model to", args.output_dir)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
