#!/bin/bash

# Configuration
ROLLS=${1:-1000}
DICE="cuboctahedron"
ENERGIES=(0.5 0.8 1.0 1.2 1.5 2.0)

echo "Starting batch simulation of $DICE with $ROLLS rolls per energy level..."

for ENERGY in "${ENERGIES[@]}"; do
    echo "=========================================================="
    echo "Running simulation with Extra Energy = $ENERGY"
    echo "=========================================================="
    ./.venv/bin/python dice_sim.py --dice "$DICE" --rolls "$ROLLS" --extra-energy "$ENERGY"
    echo ""
done

echo "Batch simulation complete."
