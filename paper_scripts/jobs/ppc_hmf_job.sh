#!/bin/bash
#SBATCH --job-name=ppc_hmf
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --output=/home/osavchenko/jax_nbody_emulator/jobs_outputs/%x-%j-%N_slurm.out
#SBATCH --error=/home/osavchenko/jax_nbody_emulator/jobs_outputs/R-%x.%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
source /home/osavchenko/jax_nbody_emulator/.venv_nbodykit/bin/activate

cd /home/osavchenko/jax_nbody_emulator

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-18}
export PYTHONNOUSERSITE=1

# HOW TO RUN (examples):
# 1) Default run:
#    sbatch paper_scripts/jobs/ppc_hmf_job.sh
# 2) Compute HMF for first 100 samples:
#    sbatch --export=ALL,NUM_SAMPLES=100 paper_scripts/jobs/ppc_hmf_job.sh
# 3) Continue from sample index 100:
#    sbatch --export=ALL,NUM_SAMPLES=100,SAMPLE_START=100 paper_scripts/jobs/ppc_hmf_job.sh
# 4) Override FoF/HMF settings:
#    sbatch --export=ALL,FOF_LINKING_LENGTH=0.2,FOF_NMIN=20,HMF_NBINS=25 paper_scripts/jobs/ppc_hmf_job.sh
# 5) Add truth-HMF overlay from displacement:
#    sbatch --export=ALL,TRUTH_DISPLACEMENT_PATH=/path/to/truth_dis_z0.npy paper_scripts/jobs/ppc_hmf_job.sh
# 6) Add truth-HMF overlay from Quijote FoF group_tab:
#    sbatch --export=ALL,TRUTH_FOF_TAB_PATH=/path/to/group_tab_004.0 paper_scripts/jobs/ppc_hmf_job.sh

SAMPLES_DIR=${SAMPLES_DIR:-/gpfs/scratch1/shared/osavchenko/mnras_paper/samples/260303_224627_net_IsotropicD}
SAMPLE_PATTERN=${SAMPLE_PATTERN:-sample_*.npy}
NUM_SAMPLES=${NUM_SAMPLES:-10}
SAMPLE_START=${SAMPLE_START:-0}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
SAMPLE_FOLDER_TEMPLATE=${SAMPLE_FOLDER_TEMPLATE:-sample_{sample_id:04d}}
RUN_NAME=${RUN_NAME:-$(basename "${SAMPLES_DIR}")}

Z_TARGET=${Z_TARGET:-0.0}
N_PART=${N_PART:-512}
BOXSIZE=${BOXSIZE:-1000}
OMEGA_M=${OMEGA_M:-0.3175}

FOF_LINKING_LENGTH=${FOF_LINKING_LENGTH:-0.2}
FOF_ABSOLUTE_LINKING=${FOF_ABSOLUTE_LINKING:-0}
FOF_NMIN=${FOF_NMIN:-20}

HMF_NBINS=${HMF_NBINS:-25}
HMF_MMIN=${HMF_MMIN:-}
HMF_MMAX=${HMF_MMAX:-}
TRUTH_DISPLACEMENT_PATH=${TRUTH_DISPLACEMENT_PATH:-}
TRUTH_FOF_TAB_PATH=${TRUTH_FOF_TAB_PATH:-/home/osavchenko/gaussian_npe/group_tab_004.0}

AGGREGATE_DIR=${AGGREGATE_DIR:-/home/osavchenko/gaussian_npe/paper_plots_scripts/${RUN_NAME}/posterior_resimulation}
AGGREGATE_PREFIX=${AGGREGATE_PREFIX:-posterior_predictive}
SAVE_PER_SAMPLE_HMF=${SAVE_PER_SAMPLE_HMF:-1}

EXTRA_ARGS=()
if [ "${FOF_ABSOLUTE_LINKING}" -eq 1 ]; then EXTRA_ARGS+=(--fof-absolute-linking); else EXTRA_ARGS+=(--no-fof-absolute-linking); fi
if [ "${SAVE_PER_SAMPLE_HMF}" -eq 1 ]; then EXTRA_ARGS+=(--save-per-sample-hmf); else EXTRA_ARGS+=(--no-save-per-sample-hmf); fi
if [ -n "${HMF_MMIN}" ]; then EXTRA_ARGS+=(--hmf-mmin "${HMF_MMIN}"); fi
if [ -n "${HMF_MMAX}" ]; then EXTRA_ARGS+=(--hmf-mmax "${HMF_MMAX}"); fi
if [ -n "${TRUTH_DISPLACEMENT_PATH}" ]; then EXTRA_ARGS+=(--truth-displacement-path "${TRUTH_DISPLACEMENT_PATH}"); fi
if [ -n "${TRUTH_FOF_TAB_PATH}" ]; then EXTRA_ARGS+=(--truth-fof-tab-path "${TRUTH_FOF_TAB_PATH}"); fi

srun python paper_scripts/ppc_hmf.py \
  --samples-dir "${SAMPLES_DIR}" \
  --sample-pattern "${SAMPLE_PATTERN}" \
  --num-samples "${NUM_SAMPLES}" \
  --sample-start "${SAMPLE_START}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  --sample-folder-template "${SAMPLE_FOLDER_TEMPLATE}" \
  --z-target "${Z_TARGET}" \
  --n-part "${N_PART}" \
  --boxsize "${BOXSIZE}" \
  --omega-m "${OMEGA_M}" \
  --fof-linking-length "${FOF_LINKING_LENGTH}" \
  --fof-nmin "${FOF_NMIN}" \
  --hmf-nbins "${HMF_NBINS}" \
  --aggregate-dir "${AGGREGATE_DIR}" \
  --aggregate-prefix "${AGGREGATE_PREFIX}" \
  "${EXTRA_ARGS[@]}"
