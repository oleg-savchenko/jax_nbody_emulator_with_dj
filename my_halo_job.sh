#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=rome
#SBATCH --time=0:40:00
#SBATCH --export=ALL
#SBATCH --output=/home/osavchenko/jax_nbody_emulator/jobs_outputs/%x-%j-%N_slurm.out
#SBATCH --error=/home/osavchenko/jax_nbody_emulator/jobs_outputs/R-%x.%j.err

module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
cd /home/osavchenko/jax_nbody_emulator

# NOTE:
# - plot-only mode does not import nbodykit and is intended for .venv (Pylians overlays).
# - full FoF mode imports nbodykit and requires .venv_nbodykit.
# - Slurm resources are static at submission time. This script sets mode-specific `srun`
#   rank/cpu usage and caps FoF ranks by allocated `SLURM_NTASKS`.
#
# HOW TO RUN:
# 1) Plot-only mode (default):
#      sbatch my_halo_job.sh
#
# 2) Full FoF generation mode:
#      sbatch --export=ALL,HALO_PLOT_ONLY=0 my_halo_job.sh
#    or explicitly by positional mode:
#      sbatch my_halo_job.sh full
#
# 3) Full FoF with custom MPI ranks (capped by allocated --ntasks):
#      sbatch --export=ALL,HALO_PLOT_ONLY=0,HALO_FOF_NTASKS=32 my_halo_job.sh
MODE_ARG=${1:-}
if [ -n "${MODE_ARG}" ]; then
  case "${MODE_ARG}" in
    full|fof|FULL|FOF|0|false|FALSE|no|NO) HALO_PLOT_ONLY=0 ;;
    plot|plot-only|PLOT|PLOT-ONLY|1|true|TRUE|yes|YES) HALO_PLOT_ONLY=1 ;;
    *)
      echo "[halo job] Unknown mode '${MODE_ARG}'. Use 'full' or 'plot' (or leave empty)." >&2
      exit 2
      ;;
  esac
else
  HALO_PLOT_ONLY=${HALO_PLOT_ONLY:-1}
fi

if [ "${HALO_PLOT_ONLY}" -eq 1 ]; then
  source /home/osavchenko/jax_nbody_emulator/.venv/bin/activate
  echo "[halo job] Mode: plot-only (--plot-only). Using .venv."
  # Plot-only relies on local/user packages in this setup (e.g. numpy/Pylians in ~/.local).
  export PYTHONNOUSERSITE=0
else
  source /home/osavchenko/jax_nbody_emulator/.venv_nbodykit/bin/activate
  echo "[halo job] Mode: full FoF generation. Using .venv_nbodykit."
  # Full FoF mode uses a self-contained env; keep user-site disabled.
  export PYTHONNOUSERSITE=1
fi

EMU_OUTPUT_DIR=${EMU_OUTPUT_DIR:-outputs/discodj_emulator_seed42_res512_box1000}
HALO_OUT_DIR=${HALO_OUT_DIR:-${EMU_OUTPUT_DIR}/halos}
LINKING_LENGTH=${LINKING_LENGTH:-0.2}
NMIN=${NMIN:-20}
SLICE_AXIS=${SLICE_AXIS:-x}
SLICE_WIDTH=${SLICE_WIDTH:-20.0}
HALO_FOF_NTASKS=${HALO_FOF_NTASKS:-16}

EXTRA_ARGS=()
if [ "${HALO_PLOT_ONLY}" -eq 1 ]; then
  RUN_NTASKS=1
  RUN_CPUS_PER_TASK=1
  EXTRA_ARGS+=(--plot-only)
  echo "[halo job] Runtime layout: ntasks=${RUN_NTASKS}, cpus-per-task=${RUN_CPUS_PER_TASK}"
else
  if [ -n "${SLURM_NTASKS:-}" ] && [ "${SLURM_NTASKS}" -lt "${HALO_FOF_NTASKS}" ]; then
    RUN_NTASKS="${SLURM_NTASKS}"
  else
    RUN_NTASKS="${HALO_FOF_NTASKS}"
  fi
  RUN_CPUS_PER_TASK=1
  echo "[halo job] Runtime layout: ntasks=${RUN_NTASKS}, cpus-per-task=${RUN_CPUS_PER_TASK} (FoF MPI)"
fi

export OMP_NUM_THREADS=${RUN_CPUS_PER_TASK}

srun --ntasks="${RUN_NTASKS}" --cpus-per-task="${RUN_CPUS_PER_TASK}" python scripts/halos.py \
  --emulator-output-dir "${EMU_OUTPUT_DIR}" \
  --output-dir "${HALO_OUT_DIR}" \
  --linking-length "${LINKING_LENGTH}" \
  --nmin "${NMIN}" \
  --slice-axis "${SLICE_AXIS}" \
  --slice-width "${SLICE_WIDTH}" \
  "${EXTRA_ARGS[@]}"
