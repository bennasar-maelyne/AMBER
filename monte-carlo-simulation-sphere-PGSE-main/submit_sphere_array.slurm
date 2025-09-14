#!/bin/bash

#SBATCH --job-name=sphere_array          # A descriptive name for the entire job array
#SBATCH --array=1-341                    # <<< IMPORTANT: Set this to your total number of experiments
#SBATCH --output=slurm_logs/sphere_%a.log  # Directs output to a unique log file for each task (%a is the task ID)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                     # Request one GPU for each experiment
#SBATCH --time=00:30:00                  # Set the time limit for each individual experiment
#SBATCH --mem=32G                        # Request a safe amount of CPU RAM for each experiment

# --- Your script starts here ---

# It's good practice to create the log directory beforehand
mkdir -p slurm_logs

echo "---"
echo "Starting Slurm Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "---"

# This is the key step to match your directory names.
# It takes the task ID (e.g., 1, 42, 341) and pads it with leading zeros to make it four digits (0001, 0042, 0341).
TASK_ID_PADDED=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
EXP_DIR="sphere_${TASK_ID_PADDED}"

# Check if the specific experiment directory actually exists
if [ -d "$EXP_DIR" ]; then
  # Navigate into the correct experiment directory
  cd "$EXP_DIR"

  echo "Successfully changed into directory: $(pwd)"
  echo "Starting simulation..."
  
  # Run the executable that is already inside this directory
  cp -a /nfs/home2/jmarchan/monte-carlo-simulation-sphere-PGSE/lib/rms/main_PGSE_bvec_cuda .
  ./main_PGSE_bvec_cuda
  
  echo "Finished simulation in $(pwd)"
else
  echo "Error: Directory ${EXP_DIR} does not exist. Skipping this task."
fi

echo "---"
echo "Slurm Array Task ${SLURM_ARRAY_TASK_ID} has finished."
echo "---"

