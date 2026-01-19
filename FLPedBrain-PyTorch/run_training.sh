#!/bin/bash
# CDS Baseline Training Script

# Create output directories
mkdir -p checkpoints
mkdir -p logs

echo "=============================================="
echo "CDS Baseline Training"
echo "=============================================="

python train.py --epochs 200

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Best model saved to: checkpoints/cds_best.pth"
