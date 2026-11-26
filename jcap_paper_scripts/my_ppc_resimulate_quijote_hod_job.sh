#!/bin/bash
# Auto-submitting launcher for jcap_paper_scripts/ppc_resimulate_quijote_hod.py

set -euo pipefail

REPO_DIR=/home/osavchenko/jax_nbody_emulator
SCRIPT_PATH=${SCRIPT_PATH:-${REPO_DIR}/jcap_paper_scripts/ppc_resimulate_quijote_hod.py}
STAGE=${STAGE:-forward}
AUTO_SUBMIT=${AUTO_SUBMIT:-1}

SBATCH_OUTPUT=${SBATCH_OUTPUT:-${REPO_DIR}/jobs_outputs/%x-%j-%N_slurm.out}
SBATCH_ERROR=${SBATCH_ERROR:-${REPO_DIR}/jobs_outputs/R-%x.%j.err}

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${AUTO_SUBMIT}" -eq 1 ]; then
  case "${STAGE}" in
    halos)
      JOB_NAME=${JOB_NAME:-ppc_qh_halos}
      CPU_PARTITION=${CPU_PARTITION:-rome}
      CPU_NODES=${CPU_NODES:-1}
      CPU_NTASKS=${CPU_NTASKS:-16}
      CPU_CPUS_PER_TASK=${CPU_CPUS_PER_TASK:-1}
      CPU_TIME=${CPU_TIME:-08:00:00}
      echo "[ppc_qh job] Submitting CPU/MPI job for STAGE=${STAGE} (${CPU_PARTITION})"
      exec sbatch \
        --job-name="${JOB_NAME}" \
        --nodes="${CPU_NODES}" \
        --ntasks="${CPU_NTASKS}" \
        --cpus-per-task="${CPU_CPUS_PER_TASK}" \
        --partition="${CPU_PARTITION}" \
        --time="${CPU_TIME}" \
        --output="${SBATCH_OUTPUT}" \
        --error="${SBATCH_ERROR}" \
        --export=ALL,AUTO_SUBMIT=0 \
        "$0"
      ;;
    hod)
      JOB_NAME=${JOB_NAME:-ppc_qh_hod}
      GPU_PARTITION=${GPU_PARTITION:-gpu_a100}
      GPU_NODES=${GPU_NODES:-1}
      GPU_NTASKS=${GPU_NTASKS:-1}
      GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK:-8}
      GPU_COUNT=${GPU_COUNT:-1}
      GPU_TIME=${GPU_TIME:-04:00:00}
      echo "[ppc_qh job] Submitting GPU job for STAGE=hod (${GPU_PARTITION})"
      exec sbatch \
        --job-name="${JOB_NAME}" \
        --nodes="${GPU_NODES}" \
        --ntasks="${GPU_NTASKS}" \
        --cpus-per-task="${GPU_CPUS_PER_TASK}" \
        --gpus="${GPU_COUNT}" \
        --partition="${GPU_PARTITION}" \
        --time="${GPU_TIME}" \
        --output="${SBATCH_OUTPUT}" \
        --error="${SBATCH_ERROR}" \
        --export=ALL,AUTO_SUBMIT=0 \
        "$0"
      ;;
    forward|all)
      JOB_NAME=${JOB_NAME:-ppc_qh_${STAGE}}
      GPU_PARTITION=${GPU_PARTITION:-gpu_h100}
      GPU_NODES=${GPU_NODES:-1}
      GPU_NTASKS=${GPU_NTASKS:-1}
      GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK:-18}
      GPU_COUNT=${GPU_COUNT:-1}
      GPU_TIME=${GPU_TIME:-01:30:00}
      echo "[ppc_qh job] Submitting GPU job for STAGE=${STAGE} (${GPU_PARTITION})"
      if [ "${STAGE}" = "all" ]; then
        echo "[ppc_qh job] STAGE=all requires an env containing emulator + nbodykit + field_level_sbi deps."
      fi
      exec sbatch \
        --job-name="${JOB_NAME}" \
        --nodes="${GPU_NODES}" \
        --ntasks="${GPU_NTASKS}" \
        --cpus-per-task="${GPU_CPUS_PER_TASK}" \
        --gpus="${GPU_COUNT}" \
        --partition="${GPU_PARTITION}" \
        --time="${GPU_TIME}" \
        --output="${SBATCH_OUTPUT}" \
        --error="${SBATCH_ERROR}" \
        --export=ALL,AUTO_SUBMIT=0 \
        "$0"
      ;;
    *)
      echo "[ppc_qh job] Unknown STAGE='${STAGE}' (use forward|halos|hod|all)" >&2
      exit 2
      ;;
  esac
fi

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
if [ "${STAGE}" = "halos" ] || [ "${STAGE}" = "hod" ] || [ "${STAGE}" = "all" ]; then
  module load OpenMPI/5.0.3-GCC-13.3.0
fi

cd "${REPO_DIR}"

VENV_PATH=${VENV_PATH:-}
HOD_VENV_PATH=${HOD_VENV_PATH:-}
HOD_USE_SYSTEM_PYTHON=${HOD_USE_SYSTEM_PYTHON:-0}
HOD_PYTHONNOUSERSITE=${HOD_PYTHONNOUSERSITE:-1}

if [ "${STAGE}" = "halos" ]; then
  source /home/osavchenko/jax_nbody_emulator/.venv_nbodykit/bin/activate
  export PYTHONNOUSERSITE=1
elif [ "${STAGE}" = "hod" ]; then
  if [ "${HOD_USE_SYSTEM_PYTHON}" -eq 1 ]; then
    echo "[ppc_qh job] STAGE=hod using module/system Python (no venv activation)"
  elif [ -n "${HOD_VENV_PATH}" ]; then
    source "${HOD_VENV_PATH}"
  elif [ -n "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
  else
    # Prefer emulator env for HOD (field_level_sbi + jax). nbodykit env often
    # lacks these modules.
    source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate
  fi
  export PYTHONNOUSERSITE="${HOD_PYTHONNOUSERSITE}"
  export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}
  export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
  export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
elif [ "${STAGE}" = "all" ]; then
  if [ -n "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
  else
    source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate
    echo "[ppc_qh job] STAGE=all defaulted to .venv. Set VENV_PATH to a combined env if needed."
  fi
  export PYTHONNOUSERSITE=0
  export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}
  export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
  export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
else
  source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate
  export PYTHONNOUSERSITE=0
  export TF_GPU_ALLOCATOR=${TF_GPU_ALLOCATOR:-cuda_malloc_async}
  export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
  export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
fi

if [ "${STAGE}" = "halos" ]; then
  RUN_NTASKS=${SLURM_NTASKS:-1}
  RUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
elif [ "${STAGE}" = "hod" ]; then
  # HOD stage is root-write by design; force single task and single GPU.
  RUN_NTASKS=1
  RUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
else
  RUN_NTASKS=1
  RUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-18}
fi

export OMP_NUM_THREADS=${RUN_CPUS_PER_TASK}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}
mkdir -p "${MPLCONFIGDIR}"

if [ "${STAGE}" = "hod" ]; then
  echo "[ppc_qh job] HOD python: $(which python)"
  if ! python - <<'PY'
import sys
import numpy as np
import jax
print(f"[ppc_qh job] Python executable: {sys.executable}")
print(f"[ppc_qh job] numpy version: {np.__version__}")
print(f"[ppc_qh job] jax version: {jax.__version__}")
devices = jax.devices()
backend = jax.default_backend()
print(f"[ppc_qh job] jax backend: {backend}")
print(f"[ppc_qh job] jax devices: {devices}")
if not any(getattr(d, "platform", "") in ("gpu", "cuda") for d in devices):
    raise RuntimeError("No CUDA GPU detected by JAX for STAGE=hod")
PY
  then
    if [ "${PYTHONNOUSERSITE}" = "1" ]; then
      echo "[ppc_qh job] numpy import failed with PYTHONNOUSERSITE=1; retrying with PYTHONNOUSERSITE=0"
      export PYTHONNOUSERSITE=0
      python - <<'PY'
import sys
import numpy as np
import jax
print(f"[ppc_qh job] Python executable: {sys.executable}")
print(f"[ppc_qh job] numpy version (usersite fallback): {np.__version__}")
print(f"[ppc_qh job] jax version (usersite fallback): {jax.__version__}")
devices = jax.devices()
backend = jax.default_backend()
print(f"[ppc_qh job] jax backend (usersite fallback): {backend}")
print(f"[ppc_qh job] jax devices (usersite fallback): {devices}")
if not any(getattr(d, "platform", "") in ("gpu", "cuda") for d in devices):
    raise RuntimeError("No CUDA GPU detected by JAX for STAGE=hod")
PY
    else
      echo "[ppc_qh job] FATAL: numpy import failed in HOD env" >&2
      exit 1
    fi
  fi
fi

IC_SAMPLES_ROOT=${IC_SAMPLES_ROOT:-/gpfs/scratch1/shared/osavchenko/jcap_paper/260531_152133_hod_onthefly_64_1ksims_60ep_logMmin13.0/ic_samples}
OUTPUT_ROOT=${OUTPUT_ROOT:-/gpfs/scratch1/shared/osavchenko/jcap_paper/260531_152133_hod_onthefly_64_1ksims_60ep_logMmin13.0/ppc}
RUN_CONFIG=${RUN_CONFIG:-/home/osavchenko/field_level_sbi/runs/260531_152133_hod_onthefly_64_1ksims_60ep_logMmin13.0/config.json}
FIELD_LEVEL_SBI_ROOT=${FIELD_LEVEL_SBI_ROOT:-/home/osavchenko/field_level_sbi}
SIM_INDEX=${SIM_INDEX:-1}

SIM_PATTERN=${SIM_PATTERN:-sim_*}
SIM_IDS=${SIM_IDS:-}
SAMPLE_PATTERN=${SAMPLE_PATTERN:-sample_*.npy}
NUM_SAMPLES=${NUM_SAMPLES:-}
SAMPLE_START=${SAMPLE_START:-0}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
INCLUDE_DELTA_IC=${INCLUDE_DELTA_IC:-1}

N_PART=${N_PART:-512}
RES=${RES:-64}
BOXSIZE=${BOXSIZE:-1000}
NDIV=${NDIV:-4,2,2}
NDIV_FALLBACKS=${NDIV_FALLBACKS:-"8,4,4;8,8,8"}
EMU_PRECISION=${EMU_PRECISION:-f16}
EMU_NUM_SIMS=${EMU_NUM_SIMS:-2}
CLASS_N_MODES=${CLASS_N_MODES:-5000}
MODE_INJECT_SEED_POLICY=${MODE_INJECT_SEED_POLICY:-sample_id}
MODE_INJECT_FIXED_SEED=${MODE_INJECT_FIXED_SEED:-0}
IC_SAMPLE_SPACE=${IC_SAMPLE_SPACE:-auto}
IC_RESCALING_FACTOR=${IC_RESCALING_FACTOR:-}
IC_AUTO_SCALE_THRESHOLD=${IC_AUTO_SCALE_THRESHOLD:-5.0}

LINKING_LENGTH=${LINKING_LENGTH:-0.2}
ABSOLUTE_LINKING=${ABSOLUTE_LINKING:-0}
NMIN=${NMIN:-20}
OMEGA_M=${OMEGA_M:-0.3175}

HMF_NBINS=${HMF_NBINS:-25}
HMF_MMIN=${HMF_MMIN:-}
HMF_MMAX=${HMF_MMAX:-}

MAS_WORDER=${MAS_WORDER:-4}
DECONVOLVE_MAS=${DECONVOLVE_MAS:-1}

REUSE_EXISTING=${REUSE_EXISTING:-1}
CLEANUP_TEMP_DISPLACEMENT=${CLEANUP_TEMP_DISPLACEMENT:-1}

SIGMA_H=${SIGMA_H:-}
H_MAX=${H_MAX:-}
PER_HOST_CAP=${PER_HOST_CAP:-}
N_MAX_SAT=${N_MAX_SAT:-}
HOD_BATCH_SIZE=${HOD_BATCH_SIZE:-8}

if [ -z "${RUN_PLOT_ANALYSIS+x}" ]; then
  if [ "${STAGE}" = "hod" ]; then
    RUN_PLOT_ANALYSIS=0
  else
    RUN_PLOT_ANALYSIS=1
  fi
fi
PLOT_OUTPUT_ROOT=${PLOT_OUTPUT_ROOT:-/home/osavchenko/field_level_sbi/paper_scripts/260531_152133_hod_onthefly_64_1ksims_60ep_logMmin13.0}
PLOT_N_WORKERS=${PLOT_N_WORKERS:-}

MANIFEST_NAME=${MANIFEST_NAME:-ppc_manifest.json}

EXTRA_ARGS=()
if [ -n "${SIM_IDS}" ]; then EXTRA_ARGS+=(--sim-ids "${SIM_IDS}"); fi
if [ -n "${NUM_SAMPLES}" ]; then EXTRA_ARGS+=(--num-samples "${NUM_SAMPLES}"); fi
if [ -n "${HMF_MMIN}" ]; then EXTRA_ARGS+=(--hmf-mmin "${HMF_MMIN}"); fi
if [ -n "${HMF_MMAX}" ]; then EXTRA_ARGS+=(--hmf-mmax "${HMF_MMAX}"); fi
if [ -n "${SIGMA_H}" ]; then EXTRA_ARGS+=(--sigma-h "${SIGMA_H}"); fi
if [ -n "${H_MAX}" ]; then EXTRA_ARGS+=(--h-max "${H_MAX}"); fi
if [ -n "${PER_HOST_CAP}" ]; then EXTRA_ARGS+=(--per-host-cap "${PER_HOST_CAP}"); fi
if [ -n "${N_MAX_SAT}" ]; then EXTRA_ARGS+=(--n-max-sat "${N_MAX_SAT}"); fi
if [ -n "${HOD_BATCH_SIZE}" ]; then EXTRA_ARGS+=(--hod-batch-size "${HOD_BATCH_SIZE}"); fi
if [ -n "${PLOT_N_WORKERS}" ]; then EXTRA_ARGS+=(--plot-n-workers "${PLOT_N_WORKERS}"); fi
if [ -n "${IC_RESCALING_FACTOR}" ]; then EXTRA_ARGS+=(--ic-rescaling-factor "${IC_RESCALING_FACTOR}"); fi

if [ "${ABSOLUTE_LINKING}" -eq 1 ]; then EXTRA_ARGS+=(--absolute-linking); else EXTRA_ARGS+=(--no-absolute-linking); fi
if [ "${DECONVOLVE_MAS}" -eq 1 ]; then EXTRA_ARGS+=(--deconvolve-mas); else EXTRA_ARGS+=(--no-deconvolve-mas); fi
if [ "${REUSE_EXISTING}" -eq 1 ]; then EXTRA_ARGS+=(--reuse-existing); else EXTRA_ARGS+=(--no-reuse-existing); fi
if [ "${CLEANUP_TEMP_DISPLACEMENT}" -eq 1 ]; then EXTRA_ARGS+=(--cleanup-temp-displacement); else EXTRA_ARGS+=(--no-cleanup-temp-displacement); fi
if [ "${RUN_PLOT_ANALYSIS}" -eq 1 ]; then EXTRA_ARGS+=(--run-plot-analysis); else EXTRA_ARGS+=(--no-run-plot-analysis); fi
if [ "${INCLUDE_DELTA_IC}" -eq 1 ]; then EXTRA_ARGS+=(--include-delta-ic); else EXTRA_ARGS+=(--no-include-delta-ic); fi

srun --ntasks="${RUN_NTASKS}" --cpus-per-task="${RUN_CPUS_PER_TASK}" \
  python "${SCRIPT_PATH}" \
  --stage "${STAGE}" \
  --ic-samples-root "${IC_SAMPLES_ROOT}" \
  --sim-index "${SIM_INDEX}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-config "${RUN_CONFIG}" \
  --field-level-sbi-root "${FIELD_LEVEL_SBI_ROOT}" \
  --sim-pattern "${SIM_PATTERN}" \
  --sample-pattern "${SAMPLE_PATTERN}" \
  --sample-start "${SAMPLE_START}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  --n-part "${N_PART}" \
  --res "${RES}" \
  --boxsize "${BOXSIZE}" \
  --ndiv "${NDIV}" \
  --ndiv-fallbacks "${NDIV_FALLBACKS}" \
  --emu-precision "${EMU_PRECISION}" \
  --emu-num-sims "${EMU_NUM_SIMS}" \
  --class-n-modes "${CLASS_N_MODES}" \
  --mode-inject-seed-policy "${MODE_INJECT_SEED_POLICY}" \
  --mode-inject-fixed-seed "${MODE_INJECT_FIXED_SEED}" \
  --ic-sample-space "${IC_SAMPLE_SPACE}" \
  --ic-auto-scale-threshold "${IC_AUTO_SCALE_THRESHOLD}" \
  --linking-length "${LINKING_LENGTH}" \
  --nmin "${NMIN}" \
  --omega-m "${OMEGA_M}" \
  --hmf-nbins "${HMF_NBINS}" \
  --mas-worder "${MAS_WORDER}" \
  --plot-output-root "${PLOT_OUTPUT_ROOT}" \
  --manifest-name "${MANIFEST_NAME}" \
  "${EXTRA_ARGS[@]}"
