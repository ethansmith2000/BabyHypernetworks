#!/bin/bash
# =============================================================================
# Dynamic FFN Parameter Sweep
#
# Sweeps over dynamic transform FFN variants with modular configuration.
#
# Usage:
#   ./launch_sweep.sh               # Run everything enabled
#   ./launch_sweep.sh baseline      # Run only baseline (swiglu)
#   ./launch_sweep.sh sweep         # Run only FFN sweep
#   ./launch_sweep.sh dry           # Dry run - show jobs without submitting
# =============================================================================

set -euo pipefail

DRY_RUN=false

SCRIPT_DIR="/home/ethan/BabyHypernetworks"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"
LOG_DIR="${SCRIPT_DIR}/logs"
CONFIG_DIR="${SCRIPT_DIR}/sweep_configs"

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}"

# ============================================================================
# DDP CONFIGURATION
# ============================================================================
NUM_GPUS=1
CPUS_PER_GPU=20

# ============================================================================
# COMMON SETTINGS
# ============================================================================

# Model configurations: "hidden_size depth n_head lr batch_size max_train_steps kron_p kron_q"
# Note: For Kronecker-based FFNs, hidden_size must equal kron_p * kron_q

# 1024 = 32 * 32
MODEL_CONFIG="1024 12 8 3.0e-4 32 200000 32 32"
# MODEL_CONFIG="1024 24 8 3.0e-4 64 200000 32 32"

# 2048 = 64 * 32 or 32 * 64
# MODEL_CONFIG="2048 12 16 3.0e-4 32 150000 64 32"

# 4096 = 64 * 64
# MODEL_CONFIG="4096 12 32 2.0e-4 32 150000 64 64"

read -r HIDDEN_SIZE DEPTH N_HEAD LR BATCH_SIZE MAX_TRAIN_STEPS KRON_P KRON_Q <<< "${MODEL_CONFIG}"

WD="0.01"
BETA2="0.999"
WARMUP_STEPS="500"

# ============================================================================
# WHICH EXPERIMENTS TO RUN
# ============================================================================
run_baseline=false           # Standard SwiGLU baseline

run_spectral=true           # SpectralModulation (as FFN replacement or third layer)
run_lowrank_hypergate=false # LowRankHyperGate (SwiGLU + dynamic perturbation)
run_lowrank_hyperffn=false  # LowRankHyperFFN (plain FFN + dynamic perturbation)
run_butterfly=false         # ButterflyTransformLayer

# ============================================================================
# SPECTRAL MODULATION SETTINGS
# ============================================================================
# Transform types: dct (real), hadamard (real, power-of-2), fft (complex, magnitude only),
#                  fft_complex (complex, both magnitude AND phase control - 2x params)
# Mask activations: none (linear), tanh (bounded), silu (unbounded), softsign (soft bounded)
# Additive branch: false (mult only), true (mult + additive frequency injection)

spectral_as_third_layer=false   # false = replace FFN, true = add as third residual after FFN
spectral_base_ffn="mlp"        # Base FFN when using as third layer: swiglu | mlp

spectral_transforms=("fft")
spectral_mask_acts=("none")
spectral_use_additives=("false")  # Additive frequency injection branch
spectral_bottlenecks=(null)     # Use (null) for direct projection

# ============================================================================
# LOWRANK HYPERGATE SETTINGS (SwiGLU-based dynamic FFN)
# ============================================================================
# Factor types: lowrank (U@V^T), kronecker (A⊗B), multiplicative (rank-1 row*col scaling)
# Delta targets: gate_input, linear_input, both_input, gate_output, linear_output, both_output, w2_output, full

hypergate_as_third_layer=false
hypergate_base_ffn="swiglu"

hypergate_factor_types=("lowrank" "multiplicative")
hypergate_ranks=(16)           # Ignored when factor_type=kronecker
hypergate_delta_targets=("gate_input" "full")

# ============================================================================
# LOWRANK HYPERFFN SETTINGS (plain FFN + dynamic perturbation)
# ============================================================================
# Factor types: lowrank (U@V^T), multiplicative (rank-1 row*col),
#               circulant (FFT conv, O(d) params), basis (K coeffs over frozen bases),
#               fastfood (S·H·G·Π·H·B, square only)
# Delta targets: w1_input, w1_output, w2_input, w2_output, full
# Base weight modes: regular (nn.Linear), lowrank_N (low-rank base), none (pure dynamic)
# Note: fastfood only supports w1_input/w2_input (square transforms), requires power-of-2 dim

hyperffn_as_third_layer=true
hyperffn_base_ffn="mlp"

hyperffn_factor_types=("multiplicative" "circulant" "basis")
hyperffn_ranks=(16)            # Also used as n_bases for basis factor_type
hyperffn_delta_targets=("w1_output")
hyperffn_base_weight_modes=("regular")

# ============================================================================
# SUBMIT JOB HELPER
# ============================================================================

submit_job() {
  local job_name="$1"
  local cfg_file="$2"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  Would submit: ${job_name}"
    echo "    Config: ${cfg_file}"
    return
  fi

  local total_cpus=$((NUM_GPUS * CPUS_PER_GPU))
  local threads_per_gpu=${CPUS_PER_GPU}
  local partition="queue1gpu"

  sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=${NUM_GPUS}
#SBATCH --cpus-per-task=${total_cpus}
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
#SBATCH --time=6-23:59:59
#SBATCH --output=${LOG_DIR}/${job_name}-%j.out
#SBATCH --error=${LOG_DIR}/${job_name}-%j.err

cd ${SCRIPT_DIR}
source /home/ethan/leo-train-template/.venv/bin/activate

export TRITON_CACHE_DIR="/home/ethan/job_triton/triton_cache_\${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/home/ethan/job_triton/inductor_cache_\${SLURM_JOB_ID}"
mkdir -p "\${TRITON_CACHE_DIR}" "\${TORCHINDUCTOR_CACHE_DIR}"

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export OMP_NUM_THREADS=${threads_per_gpu}

export MASTER_PORT=\$((29500 + (\${SLURM_JOB_ID} % 10000)))

if [[ ${NUM_GPUS} -gt 1 ]]; then
  accelerate launch \\
    --num_processes=${NUM_GPUS} \\
    --num_machines=1 \\
    --main_process_port=\${MASTER_PORT} \\
    --mixed_precision=bf16 \\
    --dynamo_backend=no \\
    ${TRAIN_SCRIPT} --override_json "${cfg_file}"
else
  python ${TRAIN_SCRIPT} --override_json "${cfg_file}"
fi
EOF

  echo "  Submitted: ${job_name}"
}

# ============================================================================
# CONFIG GENERATION HELPER
# ============================================================================

write_config() {
  local cfg_file="$1"
  local ffn_type="$2"
  local third_layer_type="$3"  # empty string or layer type
  
  # Optional params with defaults
  local lowrank_rank="${4:-16}"
  local lowrank_factor_type="${5:-lowrank}"
  local delta_target="${6:-gate_input}"
  local butterfly_rounds="${7:-3}"
  local spectral_bottleneck="${8:-null}"
  local spectral_transform="${9:-dct}"
  local spectral_mask_act="${10:-none}"
  local base_weight_mode="${11:-regular}"

  local third_layer_json="null"
  if [[ -n "${third_layer_type}" ]]; then
    third_layer_json="\"${third_layer_type}\""
  fi

  cat > "${cfg_file}" <<JSON
{
  "hidden_size": ${HIDDEN_SIZE},
  "depth": ${DEPTH},
  "n_head": ${N_HEAD},
  "learning_rate": ${LR},
  "per_device_train_batch_size": ${BATCH_SIZE},
  "max_train_steps": ${MAX_TRAIN_STEPS},
  "num_warmup_steps": ${WARMUP_STEPS},
  "weight_decay": ${WD},
  "beta2": ${BETA2},
  "ffn_type": "${ffn_type}",
  "kron_p": ${KRON_P},
  "kron_q": ${KRON_Q},
  "lowrank_rank": ${lowrank_rank},
  "lowrank_factor_type": "${lowrank_factor_type}",
  "lowrank_delta_target": "${delta_target}",
  "ffn_delta_target": "${delta_target}",
  "butterfly_rounds": ${butterfly_rounds},
  "spectral_bottleneck": ${spectral_bottleneck},
  "spectral_transform": "${spectral_transform}",
  "spectral_mask_act": "${spectral_mask_act}",
  "base_weight_mode": "${base_weight_mode}",
  "third_layer_type": ${third_layer_json}
}
JSON
}

# ============================================================================
# BASELINE
# ============================================================================

do_baseline() {
  if [[ "${run_baseline}" != "true" ]]; then
    return
  fi
  echo "=== Submitting baseline (swiglu) ==="

  local job_name="dyn-baseline-hs${HIDDEN_SIZE}-d${DEPTH}"
  local cfg_file="${CONFIG_DIR}/${job_name}.json"

  write_config "${cfg_file}" "swiglu" ""
  submit_job "${job_name}" "${cfg_file}"
}

# ============================================================================
# SPECTRAL MODULATION
# ============================================================================

do_spectral() {
  if [[ "${run_spectral}" != "true" ]]; then
    return
  fi
  echo "=== Submitting SpectralModulation experiments ==="

  local job_count=0
  
  for transform in "${spectral_transforms[@]}"; do
    for mask_act in "${spectral_mask_acts[@]}"; do
      for bn in "${spectral_bottlenecks[@]}"; do
        local bn_json="${bn}"
        [[ "${bn}" == "null" ]] || bn_json="${bn}"
        
        if [[ "${spectral_as_third_layer}" == "true" ]]; then
          # Third layer mode
          local job_name="dyn-${spectral_base_ffn}-hs${HIDDEN_SIZE}-d${DEPTH}+spectral-${transform}"
          [[ "${mask_act}" != "none" ]] && job_name="${job_name}-${mask_act}"
          [[ "${bn}" != "null" ]] && job_name="${job_name}-bn${bn}"
          
          local cfg_file="${CONFIG_DIR}/${job_name}.json"
          write_config "${cfg_file}" "${spectral_base_ffn}" "spectral" \
            16 lowrank gate_input 3 "${bn_json}" "${transform}" "${mask_act}" regular
        else
          # FFN replacement mode
          local job_name="dyn-spectral-hs${HIDDEN_SIZE}-d${DEPTH}-${transform}"
          [[ "${mask_act}" != "none" ]] && job_name="${job_name}-${mask_act}"
          [[ "${bn}" != "null" ]] && job_name="${job_name}-bn${bn}"
          
          local cfg_file="${CONFIG_DIR}/${job_name}.json"
          write_config "${cfg_file}" "spectral" "" \
            16 lowrank gate_input 3 "${bn_json}" "${transform}" "${mask_act}" regular
        fi
        
        submit_job "${job_name}" "${cfg_file}"
        job_count=$((job_count + 1))
      done
    done
  done
  
  echo "  Submitted ${job_count} spectral jobs"
}

# ============================================================================
# LOWRANK HYPERGATE (SwiGLU-based)
# ============================================================================

do_lowrank_hypergate() {
  if [[ "${run_lowrank_hypergate}" != "true" ]]; then
    return
  fi
  echo "=== Submitting LowRankHyperGate experiments ==="

  local job_count=0
  
  for factor_type in "${hypergate_factor_types[@]}"; do
    for rank in "${hypergate_ranks[@]}"; do
      for delta in "${hypergate_delta_targets[@]}"; do
        
        if [[ "${hypergate_as_third_layer}" == "true" ]]; then
          # Third layer mode
          local job_name="dyn-${hypergate_base_ffn}-hs${HIDDEN_SIZE}-d${DEPTH}+hypergate"
          if [[ "${factor_type}" == "kronecker" ]]; then
            job_name="${job_name}-kron"
          else
            job_name="${job_name}-r${rank}"
          fi
          job_name="${job_name}-${delta}"
          
          local cfg_file="${CONFIG_DIR}/${job_name}.json"
          write_config "${cfg_file}" "${hypergate_base_ffn}" "lowrank_hypergate" \
            "${rank}" "${factor_type}" "${delta}" 3 null dct none regular
        else
          # FFN replacement mode
          local job_name="dyn-lowrank_hypergate-hs${HIDDEN_SIZE}-d${DEPTH}"
          if [[ "${factor_type}" == "kronecker" ]]; then
            job_name="${job_name}-kron"
          else
            job_name="${job_name}-r${rank}"
          fi
          job_name="${job_name}-${delta}"
          
          local cfg_file="${CONFIG_DIR}/${job_name}.json"
          write_config "${cfg_file}" "lowrank_hypergate" "" \
            "${rank}" "${factor_type}" "${delta}" 3 null dct none regular
        fi
        
        submit_job "${job_name}" "${cfg_file}"
        job_count=$((job_count + 1))
      done
    done
  done
  
  echo "  Submitted ${job_count} lowrank_hypergate jobs"
}

# ============================================================================
# LOWRANK HYPERFFN (plain FFN)
# ============================================================================

do_lowrank_hyperffn() {
  if [[ "${run_lowrank_hyperffn}" != "true" ]]; then
    return
  fi
  echo "=== Submitting LowRankHyperFFN experiments ==="

  local job_count=0
  
  for factor_type in "${hyperffn_factor_types[@]}"; do
    for rank in "${hyperffn_ranks[@]}"; do
      for delta in "${hyperffn_delta_targets[@]}"; do
        for base_mode in "${hyperffn_base_weight_modes[@]}"; do
          
          if [[ "${hyperffn_as_third_layer}" == "true" ]]; then
            # Third layer mode
            local job_name="dyn-${hyperffn_base_ffn}-hs${HIDDEN_SIZE}-d${DEPTH}+hyperffn"
            if [[ "${factor_type}" == "kronecker" ]]; then
              job_name="${job_name}-kron"
            else
              job_name="${job_name}-r${rank}"
            fi
            job_name="${job_name}-${delta}"
            [[ "${base_mode}" != "regular" ]] && job_name="${job_name}-${base_mode}"
            
            local cfg_file="${CONFIG_DIR}/${job_name}.json"
            write_config "${cfg_file}" "${hyperffn_base_ffn}" "lowrank_hyperffn" \
              "${rank}" "${factor_type}" "${delta}" 3 null dct none "${base_mode}"
          else
            # FFN replacement mode
            local job_name="dyn-lowrank_hyperffn-hs${HIDDEN_SIZE}-d${DEPTH}"
            if [[ "${factor_type}" == "kronecker" ]]; then
              job_name="${job_name}-kron"
            else
              job_name="${job_name}-r${rank}"
            fi
            job_name="${job_name}-${delta}"
            [[ "${base_mode}" != "regular" ]] && job_name="${job_name}-${base_mode}"
            
            local cfg_file="${CONFIG_DIR}/${job_name}.json"
            write_config "${cfg_file}" "lowrank_hyperffn" "" \
              "${rank}" "${factor_type}" "${delta}" 3 null dct none "${base_mode}"
          fi
          
          submit_job "${job_name}" "${cfg_file}"
          job_count=$((job_count + 1))
        done
      done
    done
  done
  
  echo "  Submitted ${job_count} lowrank_hyperffn jobs"
}

# ============================================================================
# BUTTERFLY TRANSFORM
# ============================================================================

do_butterfly() {
  if [[ "${run_butterfly}" != "true" ]]; then
    return
  fi
  echo "=== Submitting Butterfly experiments ==="

  local job_count=0
  
  for rounds in "${butterfly_rounds[@]}"; do
    
    if [[ "${butterfly_as_third_layer}" == "true" ]]; then
      # Third layer mode
      local job_name="dyn-${butterfly_base_ffn}-hs${HIDDEN_SIZE}-d${DEPTH}+butterfly-r${rounds}"
      
      local cfg_file="${CONFIG_DIR}/${job_name}.json"
      write_config "${cfg_file}" "${butterfly_base_ffn}" "butterfly" \
        16 lowrank gate_input "${rounds}" null dct none regular
    else
      # FFN replacement mode
      local job_name="dyn-butterfly-hs${HIDDEN_SIZE}-d${DEPTH}-r${rounds}"
      
      local cfg_file="${CONFIG_DIR}/${job_name}.json"
      write_config "${cfg_file}" "butterfly" "" \
        16 lowrank gate_input "${rounds}" null dct none regular
    fi
    
    submit_job "${job_name}" "${cfg_file}"
    job_count=$((job_count + 1))
  done
  
  echo "  Submitted ${job_count} butterfly jobs"
}

# ============================================================================
# MAIN
# ============================================================================

run_all_sweeps() {
  do_spectral
  do_lowrank_hypergate
  do_lowrank_hyperffn
  do_butterfly
}

case "${1:-all}" in
  baseline)
    do_baseline
    ;;
  sweep)
    run_all_sweeps
    ;;
  dry)
    DRY_RUN=true
    echo "=== DRY RUN - showing what would be submitted ==="
    do_baseline
    run_all_sweeps
    ;;
  all|*)
    do_baseline
    run_all_sweeps
    ;;
esac

echo "============================================"
echo "Done!"
echo "============================================"
