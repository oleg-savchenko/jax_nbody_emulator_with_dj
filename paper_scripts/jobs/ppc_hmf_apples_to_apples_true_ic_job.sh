#!/bin/bash
#SBATCH --job-name=ppc_hmf_a2a_true_ic
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=rome
#SBATCH --time=02:00:00
#SBATCH --output=/home/osavchenko/jax_nbody_emulator/jobs_outputs/%x-%j-%N_slurm.out
#SBATCH --error=/home/osavchenko/jax_nbody_emulator/jobs_outputs/R-%x.%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
source /home/osavchenko/jax_nbody_emulator/.venv_nbodykit/bin/activate

cd /home/osavchenko/jax_nbody_emulator

# Enable emulator deps while running under nbodykit env.
export PYTHONPATH="/home/osavchenko/.local/lib/python3.12/site-packages:/home/osavchenko/jax_nbody_emulator/.venv/lib/python3.12/site-packages:/home/osavchenko/jax_nbody_emulator/src:${PYTHONPATH}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONNOUSERSITE=0
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}
mkdir -p "${MPLCONFIGDIR}"

# HOW TO RUN (examples):
# 1) Default apples-to-apples run:
#    sbatch paper_scripts/jobs/ppc_hmf_apples_to_apples_true_ic_job.sh
# 2) Change output prefix (keeps old plots untouched):
#    sbatch --export=ALL,OUTPUT_PREFIX=posterior_predictive_hmf_apples_to_apples_true_ic_comparison_v2 paper_scripts/jobs/ppc_hmf_apples_to_apples_true_ic_job.sh
# 3) Disable reuse of saved true forward field:
#    sbatch --export=ALL,REUSE_EXISTING_FORWARD=0 paper_scripts/jobs/ppc_hmf_apples_to_apples_true_ic_job.sh

SAMPLES_DIR=${SAMPLES_DIR:-/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD}
POSTERIOR_ARRAYS_PATH=${POSTERIOR_ARRAYS_PATH:-/home/osavchenko/gaussian_npe/paper_plots_scripts/260303_224627_net_IsotropicD/posterior_resimulation/posterior_predictive_hmf_arrays.npz}
TARGET_PATH=${TARGET_PATH:-/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt}
IC_KEY=${IC_KEY:-delta_z127}

Z_IC=${Z_IC:-127.0}
Z_TARGET=${Z_TARGET:-0.0}
IC_RESCALE_MODE=${IC_RESCALE_MODE:-growth}

N_PART=${N_PART:-512}
RES=${RES:-128}
BOXSIZE=${BOXSIZE:-1000}
NDIV=${NDIV:-4,2,2}
EMU_PRECISION=${EMU_PRECISION:-f16}
NUM_SIMS=${NUM_SIMS:-1}
MAS_WORDER=${MAS_WORDER:-4}
DECONVOLVE_MAS=${DECONVOLVE_MAS:-1}
DELTA_UPSAMPLE_METHOD=${DELTA_UPSAMPLE_METHOD:-mode_inject}
DOWNSAMPLE_METHOD=${DOWNSAMPLE_METHOD:-fourier_truncation}
DOWNSAMPLE_THREADS=${DOWNSAMPLE_THREADS:-1}
CLASS_N_MODES=${CLASS_N_MODES:-4000}

OMEGA_M=${OMEGA_M:-0.3175}
FOF_LINKING_LENGTH=${FOF_LINKING_LENGTH:-0.2}
FOF_ABSOLUTE_LINKING=${FOF_ABSOLUTE_LINKING:-0}
FOF_NMIN=${FOF_NMIN:-20}

OUTPUT_DIR=${OUTPUT_DIR:-/home/osavchenko/gaussian_npe/paper_plots_scripts/260303_224627_net_IsotropicD}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-posterior_predictive_hmf_apples_to_apples_true_ic_comparison}
TRUTH_FORWARD_DIR=${TRUTH_FORWARD_DIR:-${OUTPUT_DIR}/true_ic_forward_apples_to_apples}
REUSE_EXISTING_FORWARD=${REUSE_EXISTING_FORWARD:-1}
SHOW_TINKER=${SHOW_TINKER:-1}
SHOW_ORIGINAL_TRUTH=${SHOW_ORIGINAL_TRUTH:-0}

EXTRA_ARGS=()
if [ "${DECONVOLVE_MAS}" -eq 1 ]; then EXTRA_ARGS+=(--deconvolve-mas); else EXTRA_ARGS+=(--no-deconvolve-mas); fi
if [ "${FOF_ABSOLUTE_LINKING}" -eq 1 ]; then EXTRA_ARGS+=(--fof-absolute-linking); else EXTRA_ARGS+=(--no-fof-absolute-linking); fi
if [ "${REUSE_EXISTING_FORWARD}" -eq 1 ]; then EXTRA_ARGS+=(--reuse-existing-forward); else EXTRA_ARGS+=(--no-reuse-existing-forward); fi
if [ "${SHOW_TINKER}" -eq 1 ]; then EXTRA_ARGS+=(--show-tinker); else EXTRA_ARGS+=(--no-show-tinker); fi
if [ "${SHOW_ORIGINAL_TRUTH}" -eq 1 ]; then EXTRA_ARGS+=(--show-original-truth); else EXTRA_ARGS+=(--no-show-original-truth); fi

srun python paper_scripts/ppc_hmf_apples_to_apples_true_ic.py \
  --target-path "${TARGET_PATH}" \
  --ic-key "${IC_KEY}" \
  --z-ic "${Z_IC}" \
  --z-target "${Z_TARGET}" \
  --ic-rescale-mode "${IC_RESCALE_MODE}" \
  --samples-dir "${SAMPLES_DIR}" \
  --posterior-arrays-path "${POSTERIOR_ARRAYS_PATH}" \
  --n-part "${N_PART}" \
  --res "${RES}" \
  --boxsize "${BOXSIZE}" \
  --ndiv "${NDIV}" \
  --emu-precision "${EMU_PRECISION}" \
  --num-sims "${NUM_SIMS}" \
  --mas-worder "${MAS_WORDER}" \
  --delta-upsample-method "${DELTA_UPSAMPLE_METHOD}" \
  --downsample-method "${DOWNSAMPLE_METHOD}" \
  --downsample-threads "${DOWNSAMPLE_THREADS}" \
  --class-n-modes "${CLASS_N_MODES}" \
  --omega-m "${OMEGA_M}" \
  --fof-linking-length "${FOF_LINKING_LENGTH}" \
  --fof-nmin "${FOF_NMIN}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --truth-forward-dir "${TRUTH_FORWARD_DIR}" \
  "${EXTRA_ARGS[@]}"
