#!/bin/bash
# FL Training Script - Roughly matches TF approach
# To accomodate pytorch friendly pipelines
# Phase 1: Warm start on ST+SE (25 to 50 rounds) - 2 GPUs (1 site per GPU)
# Phase 2: Full FL on all 16 sites (50-100 rounds) - 8 GPUs (2 sites per GPU)
#

set -e

# Configuration
WARMSTART_ROUNDS=25
WARMSTART_SITES="ST,SE"
WARMSTART_GPUS=2
FULL_FL_ROUNDS=100
NUM_GPUS=8
LOCAL_EPOCHS=1
BATCH_SIZE=4
BATCH_SIZE_EVAL=8
NUM_WORKERS=4
LR=1e-4
LR_AFTER_WARMSTART=1e-5  # 10x lower after warm start (like TF)
DICE_WEIGHT=0.5
SAVE_EVERY=10
NO_PRETRAINED=""
FREEZE_ENCODER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --warmstart-rounds) WARMSTART_ROUNDS="$2"; shift 2;;
        --full-fl-rounds) FULL_FL_ROUNDS="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --no-pretrained) NO_PRETRAINED="--no-pretrained"; shift;;
        --freeze-encoder) FREEZE_ENCODER="--freeze-encoder"; shift;;
        --skip-warmstart) SKIP_WARMSTART=1; shift;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# Create directories
mkdir -p checkpoints/fl_weights
mkdir -p logs

echo "=============================================="
echo "FL Training (TF-style with Warm Start)"
echo "=============================================="
echo "Phase 1: Warm start on ${WARMSTART_SITES} (${WARMSTART_ROUNDS} rounds, ${WARMSTART_GPUS} GPUs)"
echo "Phase 2: Full FL on all 16 sites (${FULL_FL_ROUNDS} rounds, ${NUM_GPUS} GPUs)"
echo "Learning rate: ${LR} -> ${LR_AFTER_WARMSTART} after warm start"
echo "=============================================="

# Clean up any previous weights
rm -f checkpoints/fl_weights/*.pth
rm -f checkpoints/fl_weights/*.txt

#######################################
# Phase 1: Warm Start on ST + SE (2 GPUs)
#######################################
if [ -z "$SKIP_WARMSTART" ]; then
    echo ""
    echo "=============================================="
    echo "Phase 1: Warm Start (${WARMSTART_SITES}) - ${WARMSTART_GPUS} GPUs"
    echo "=============================================="

    # Launch 2 processes for warm start (1 site per GPU)
    PIDS=()

    for GPU_ID in $(seq 0 $((WARMSTART_GPUS - 1))); do
        echo "Launching warm start GPU ${GPU_ID}..."

        CUDA_VISIBLE_DEVICES=${GPU_ID} python train_fl_single.py \
            --gpu-id ${GPU_ID} \
            --num-gpus ${WARMSTART_GPUS} \
            --sites ${WARMSTART_SITES} \
            --batch-size ${BATCH_SIZE} \
            --batch-size-eval ${BATCH_SIZE_EVAL} \
            --num-workers ${NUM_WORKERS} \
            --num-rounds ${WARMSTART_ROUNDS} \
            --local-epochs ${LOCAL_EPOCHS} \
            --lr ${LR} \
            --dice-weight ${DICE_WEIGHT} \
            --save-every ${SAVE_EVERY} \
            ${NO_PRETRAINED} ${FREEZE_ENCODER} \
            > logs/fl_warmstart_gpu${GPU_ID}.log 2>&1 &

        PIDS+=($!)
        sleep 1
    done

    echo "Warm start processes launched. PIDs: ${PIDS[@]}"
    echo "Monitor with: tail -f logs/fl_warmstart_gpu0.log"

    # Wait for warm start to complete
    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait ${PIDS[$i]} || {
            echo "ERROR: Warm start GPU $i failed!"
            FAILED=1
        }
    done

    if [ $FAILED -ne 0 ]; then
        echo "Warm start FAILED - check logs"
        exit 1
    fi

    # Copy warm start model
    cp checkpoints/fl_best.pth checkpoints/fl_warmstart_best.pth
    echo "Warm start complete. Best model saved to checkpoints/fl_warmstart_best.pth"
fi

#######################################
# Phase 2: Full FL on all 16 sites
#######################################
echo ""
echo "=============================================="
echo "Phase 2: Full FL (all 16 sites, ${FULL_FL_ROUNDS} rounds, ${NUM_GPUS} GPUs)"
echo "=============================================="

# Clean intermediate weights
rm -f checkpoints/fl_weights/*.pth
rm -f checkpoints/fl_weights/*.txt

# Launch 8 processes in background
PIDS=()

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching GPU ${GPU_ID}..."

    RESUME_ARG=""
    if [ $GPU_ID -eq 0 ] && [ -f checkpoints/fl_warmstart_best.pth ]; then
        RESUME_ARG="--resume checkpoints/fl_warmstart_best.pth"
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python train_fl_single.py \
        --gpu-id ${GPU_ID} \
        --num-gpus ${NUM_GPUS} \
        --batch-size ${BATCH_SIZE} \
        --batch-size-eval ${BATCH_SIZE_EVAL} \
        --num-workers ${NUM_WORKERS} \
        --num-rounds ${FULL_FL_ROUNDS} \
        --local-epochs ${LOCAL_EPOCHS} \
        --lr ${LR_AFTER_WARMSTART} \
        --dice-weight ${DICE_WEIGHT} \
        --save-every ${SAVE_EVERY} \
        ${NO_PRETRAINED} ${FREEZE_ENCODER} \
        ${RESUME_ARG} \
        > logs/fl_gpu${GPU_ID}.log 2>&1 &

    PIDS+=($!)
    sleep 2
done

echo ""
echo "All ${NUM_GPUS} processes launched."
echo "PIDs: ${PIDS[@]}"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/fl_gpu0.log  # Main process with validation"
echo ""
echo "Waiting for training to complete..."

# Wait for all processes
FAILED=0
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]} || {
        echo "ERROR: GPU $i process (PID ${PIDS[$i]}) failed!"
        FAILED=1
    }
done

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "FL Training complete!"
    echo "=============================================="
    echo "Warm start model: checkpoints/fl_warmstart_best.pth"
    echo "Final best model: checkpoints/fl_best.pth"
    echo "Logs: logs/fl_*.log"
else
    echo ""
    echo "=============================================="
    echo "FL Training FAILED - check logs for errors"
    echo "=============================================="
    exit 1
fi
