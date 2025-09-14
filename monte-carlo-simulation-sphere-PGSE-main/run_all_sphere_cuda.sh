#!/bin/bash

# === Configuration ===
CUDA_EXECUTABLE_PATH="/chemin/vers/main_PGSE_bvec_cuda"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# === Loop on every directory sphere_#### ===
for dir in sphere_????; do
    echo "--------------------------------------------"
    echo " Treating directory : $dir"
    
    if [ -d "$dir" ]; then
        cp "$CUDA_EXECUTABLE_PATH" "$dir"
        cd "$dir"
        
        echo "Running simulation in $dir"
        ./main_PGSE_bvec_cuda > "../$LOG_DIR/${dir}.log" 2>&1
        
        echo "Simulation finished for $dir"
        cd ..
    else
        echo " Directory $dir not found, skipping the task."
    fi
done

echo " All simulations terminated !"
