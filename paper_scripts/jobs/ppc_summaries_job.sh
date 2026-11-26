#!/bin/bash
#SBATCH --job-name=ppc_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=02:00:00
#SBATCH --output=/home/osavchenko/jax_nbody_emulator/jobs_outputs/%x-%j-%N_slurm.out
#SBATCH --error=/home/osavchenko/jax_nbody_emulator/jobs_outputs/R-%x.%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate

cd /home/osavchenko/jax_nbody_emulator

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-18}
export PYTHONNOUSERSITE=0

# HOW TO RUN (examples):
# 1) Default run:
#    sbatch paper_scripts/jobs/ppc_summaries_job.sh
# 2) Summarize first 100 samples:
#    sbatch --export=ALL,NUM_SAMPLES=100 paper_scripts/jobs/ppc_summaries_job.sh
# 3) Summarize a shifted chunk:
#    sbatch --export=ALL,NUM_SAMPLES=100,SAMPLE_START=100 paper_scripts/jobs/ppc_summaries_job.sh
# 4) Override output folder / prefix:
#    sbatch --export=ALL,AGGREGATE_DIR=/path/to/out,AGGREGATE_PREFIX=my_run paper_scripts/jobs/ppc_summaries_job.sh

SAMPLES_DIR=${SAMPLES_DIR:-/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD}
SAMPLE_PATTERN=${SAMPLE_PATTERN:-sample_*.npy}
NUM_SAMPLES=${NUM_SAMPLES:-100}
SAMPLE_START=${SAMPLE_START:-0}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
SAMPLE_FOLDER_TEMPLATE=${SAMPLE_FOLDER_TEMPLATE:-sample_{sample_id:04d}}
RUN_NAME=${RUN_NAME:-$(basename "${SAMPLES_DIR}")}

TARGET_PATH=${TARGET_PATH:-/home/osavchenko/Quijote/Quijote_target/Quijote_sample0_wout_MAK.pt}
TARGET_KEY=${TARGET_KEY:-delta_z0}
TARGET_DECONVOLVED=${TARGET_DECONVOLVED:-1}

Z_TARGET=${Z_TARGET:-0.0}
RES=${RES:-128}
BOXSIZE=${BOXSIZE:-1000}
MAS_WORDER=${MAS_WORDER:-4}
DECONVOLVE_MAS=${DECONVOLVE_MAS:-1}
DELTA_UPSAMPLE_METHOD=${DELTA_UPSAMPLE_METHOD:-mode_inject}
PK_THREADS=${PK_THREADS:-1}

BISPEC_K1=${BISPEC_K1:-0.1}
BISPEC_K2=${BISPEC_K2:-0.1}
BISPEC_K1B=${BISPEC_K1B:-0.05}
BISPEC_K2B=${BISPEC_K2B:-0.1}
BISPEC_N_THETA=${BISPEC_N_THETA:-25}

MINKOWSKI_NU_MIN=${MINKOWSKI_NU_MIN:--3.0}
MINKOWSKI_NU_MAX=${MINKOWSKI_NU_MAX:-3.0}
MINKOWSKI_N_THRESHOLDS=${MINKOWSKI_N_THRESHOLDS:-41}

AGGREGATE_DIR=${AGGREGATE_DIR:-/home/osavchenko/gaussian_npe/paper_plots_scripts/${RUN_NAME}/posterior_resimulation}
AGGREGATE_PREFIX=${AGGREGATE_PREFIX:-posterior_predictive}
SAVE_PER_SAMPLE_SUMMARY=${SAVE_PER_SAMPLE_SUMMARY:-1}

EXTRA_ARGS=()
if [ "${TARGET_DECONVOLVED}" -eq 1 ]; then EXTRA_ARGS+=(--target-deconvolved); else EXTRA_ARGS+=(--no-target-deconvolved); fi
if [ "${DECONVOLVE_MAS}" -eq 1 ]; then EXTRA_ARGS+=(--deconvolve-mas); else EXTRA_ARGS+=(--no-deconvolve-mas); fi
if [ "${SAVE_PER_SAMPLE_SUMMARY}" -eq 1 ]; then EXTRA_ARGS+=(--save-per-sample-summary); else EXTRA_ARGS+=(--no-save-per-sample-summary); fi

srun python paper_scripts/ppc_summaries.py \
  --samples-dir "${SAMPLES_DIR}" \
  --sample-pattern "${SAMPLE_PATTERN}" \
  --num-samples "${NUM_SAMPLES}" \
  --sample-start "${SAMPLE_START}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  --sample-folder-template "${SAMPLE_FOLDER_TEMPLATE}" \
  --target-path "${TARGET_PATH}" \
  --target-key "${TARGET_KEY}" \
  --z-target "${Z_TARGET}" \
  --res "${RES}" \
  --boxsize "${BOXSIZE}" \
  --mas-worder "${MAS_WORDER}" \
  --delta-upsample-method "${DELTA_UPSAMPLE_METHOD}" \
  --pk-threads "${PK_THREADS}" \
  --bispec-k1 "${BISPEC_K1}" \
  --bispec-k2 "${BISPEC_K2}" \
  --bispec-k1b "${BISPEC_K1B}" \
  --bispec-k2b "${BISPEC_K2B}" \
  --bispec-n-theta "${BISPEC_N_THETA}" \
  --minkowski-nu-min "${MINKOWSKI_NU_MIN}" \
  --minkowski-nu-max "${MINKOWSKI_NU_MAX}" \
  --minkowski-n-thresholds "${MINKOWSKI_N_THRESHOLDS}" \
  --aggregate-dir "${AGGREGATE_DIR}" \
  --aggregate-prefix "${AGGREGATE_PREFIX}" \
  "${EXTRA_ARGS[@]}"
