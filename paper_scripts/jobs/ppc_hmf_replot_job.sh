#!/bin/bash
#SBATCH --job-name=ppc_hmf_replot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=rome
#SBATCH --time=00:10:00
#SBATCH --output=/home/osavchenko/jax_nbody_emulator/jobs_outputs/%x-%j-%N_slurm.out
#SBATCH --error=/home/osavchenko/jax_nbody_emulator/jobs_outputs/R-%x.%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0

# Replot-only flow: use module Python by default.
# Set USE_VENV=1 if you intentionally want /home/osavchenko/jax_nbody_emulator/.venv.
if [ "${USE_VENV:-0}" -eq 1 ]; then
  source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate
fi

cd /home/osavchenko/jax_nbody_emulator

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
if [ "${USE_VENV:-0}" -eq 1 ]; then
  export PYTHONNOUSERSITE=1
else
  unset PYTHONNOUSERSITE
fi
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}
mkdir -p "${MPLCONFIGDIR}"

# HOW TO RUN (examples):
# 1) Default fast replot (uses saved arrays only):
#    sbatch paper_scripts/jobs/ppc_hmf_replot_job.sh
# 2) Replot another run:
#    sbatch --export=ALL,ARRAYS_PATH=/path/to/posterior_predictive_hmf_arrays.npz paper_scripts/jobs/ppc_hmf_replot_job.sh
# 3) Add/override Tinker from an existing CLASS P(k) table:
#    sbatch --export=ALL,CLASS_PK_TABLE_PATH=/path/to/class_linear_pk_z0_table.txt,OMEGA_M=0.3175 paper_scripts/jobs/ppc_hmf_replot_job.sh
# 4) Build fiducial Quijote CLASS P(k) on the fly (no pre-existing table needed):
#    sbatch --export=ALL,BUILD_CLASS_PK_FIDUCIAL=1,OMEGA_M=0.3175 paper_scripts/jobs/ppc_hmf_replot_job.sh
# 5) Save generated fiducial CLASS table and updated arrays:
#    sbatch --export=ALL,BUILD_CLASS_PK_FIDUCIAL=1,GENERATED_CLASS_PK_SAVE_PATH=/path/to/class_linear_pk_z0_table.txt,SAVE_ARRAYS_PATH=/path/to/posterior_predictive_hmf_arrays_with_tinker.npz paper_scripts/jobs/ppc_hmf_replot_job.sh

ARRAYS_PATH=${ARRAYS_PATH:-/gpfs/home5/osavchenko/gaussian_npe/paper_plots_scripts/260303_224627_net_IsotropicD/posterior_resimulation/posterior_predictive_hmf_arrays.npz}
OUTPUT_PATH=${OUTPUT_PATH:-/gpfs/home5/osavchenko/gaussian_npe/paper_plots_scripts/260303_224627_net_IsotropicD/posterior_resimulation/posterior_predictive_hmf.png}
SAVE_ARRAYS_PATH=${SAVE_ARRAYS_PATH:-}

TITLE=${TITLE:-Friend-of-Friends mass}
RATIO_YMIN=${RATIO_YMIN:-0.6}
RATIO_YMAX=${RATIO_YMAX:-1.4}
DPI=${DPI:-180}
SHOW_TRUTH=${SHOW_TRUTH:-1}
SHOW_TINKER=${SHOW_TINKER:-1}

CLASS_PK_TABLE_PATH=${CLASS_PK_TABLE_PATH:-}
BUILD_CLASS_PK_FIDUCIAL=${BUILD_CLASS_PK_FIDUCIAL:-0}
CLASS_Z=${CLASS_Z:-0.0}
CLASS_KMIN=${CLASS_KMIN:-1e-5}
CLASS_KMAX=${CLASS_KMAX:-10.0}
CLASS_N_MODES=${CLASS_N_MODES:-4000}
GENERATED_CLASS_PK_SAVE_PATH=${GENERATED_CLASS_PK_SAVE_PATH:-}

OMEGA_M=${OMEGA_M:-0.3175}
TINKER_AUTHOR=${TINKER_AUTHOR:-Tinker}
TINKER_INTEGRATION_BINS=${TINKER_INTEGRATION_BINS:-1000}

EXTRA_ARGS=()
if [ "${SHOW_TRUTH}" -eq 1 ]; then EXTRA_ARGS+=(--show-truth); else EXTRA_ARGS+=(--no-show-truth); fi
if [ "${SHOW_TINKER}" -eq 1 ]; then EXTRA_ARGS+=(--show-tinker); else EXTRA_ARGS+=(--no-show-tinker); fi

if [ -n "${CLASS_PK_TABLE_PATH}" ] && [ "${BUILD_CLASS_PK_FIDUCIAL}" -eq 1 ]; then
  echo "Error: set either CLASS_PK_TABLE_PATH or BUILD_CLASS_PK_FIDUCIAL=1, not both." >&2
  exit 2
fi

if [ -n "${CLASS_PK_TABLE_PATH}" ]; then
  EXTRA_ARGS+=(--class-pk-table-path "${CLASS_PK_TABLE_PATH}")
elif [ "${BUILD_CLASS_PK_FIDUCIAL}" -eq 1 ]; then
  EXTRA_ARGS+=(
    --build-class-pk-fiducial
    --class-z "${CLASS_Z}"
    --class-kmin "${CLASS_KMIN}"
    --class-kmax "${CLASS_KMAX}"
    --class-n-modes "${CLASS_N_MODES}"
  )
  if [ -n "${GENERATED_CLASS_PK_SAVE_PATH}" ]; then
    EXTRA_ARGS+=(--generated-class-pk-save-path "${GENERATED_CLASS_PK_SAVE_PATH}")
  fi
fi

if [ -n "${CLASS_PK_TABLE_PATH}" ] || [ "${BUILD_CLASS_PK_FIDUCIAL}" -eq 1 ]; then
  EXTRA_ARGS+=(
    --omega-m "${OMEGA_M}"
    --tinker-author "${TINKER_AUTHOR}"
    --tinker-integration-bins "${TINKER_INTEGRATION_BINS}"
  )
fi

if [ -n "${SAVE_ARRAYS_PATH}" ]; then
  EXTRA_ARGS+=(--save-arrays-path "${SAVE_ARRAYS_PATH}")
fi

srun python paper_scripts/ppc_hmf_replot.py \
  --arrays-path "${ARRAYS_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --title "${TITLE}" \
  --ratio-ymin "${RATIO_YMIN}" \
  --ratio-ymax "${RATIO_YMAX}" \
  --dpi "${DPI}" \
  "${EXTRA_ARGS[@]}"
