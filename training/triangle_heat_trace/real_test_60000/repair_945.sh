#!/bin/sh
# Reproduce the targeted solve of dataset row 945 from its baseline checkpoint.
set -eu

if [ "$#" -ne 3 ]
then
    echo "usage: sh repair_945.sh DATASET BASELINE_945_NPZ OUTPUT_DIR" >&2
    exit 2
fi

repair_945_solver_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repair_945_python=${TRIANGLE_SPECTRUM_PYTHON:-python3}
repair_945_dataset=$1
repair_945_baseline=$2
repair_945_output=$3
mkdir -p "$repair_945_output"
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$repair_945_python" "$repair_945_solver_dir/scan_robust.py" \
    "$repair_945_dataset" "$repair_945_output/945_narrow_cut.npz" \
    --index 945 --stop 445 --step .17 --phase .31 --pad 14

"$repair_945_python" "$repair_945_solver_dir/repair_checkpoint.py" \
    "$repair_945_output/945_narrow_cut.npz" "$repair_945_output/945_narrow_repaired.npz" \
    --bands 132:137 160:166 327:343 424:438 --step .105 --phase .57 \
    --provenance "$repair_945_output/945_low_repair_provenance.json"

"$repair_945_python" "$repair_945_solver_dir/repair_checkpoint.py" \
    "$repair_945_baseline" "$repair_945_output/945_repaired.npz" \
    --low-scan "$repair_945_output/945_narrow_repaired.npz" \
    --bands 492:500 515:523 551:567 589:597 611:634 747:759 786:802 824:834 847:854 \
    --step .16 --phase .37 --provenance "$repair_945_output/945_repair_provenance.json"

"$repair_945_python" "$repair_945_solver_dir/repair_checkpoint.py" \
    "$repair_945_output/945_repaired.npz" "$repair_945_output/945_stage2.npz" \
    --bands 789:795 --step .081 --phase .73 \
    --provenance "$repair_945_output/945_high_repair_provenance.json"

"$repair_945_python" "$repair_945_solver_dir/mps_probe.py" \
    "$repair_945_output/945_repaired.npz" "$repair_945_output/945_tension_162.json" \
    162.526 162.612 --step .001

"$repair_945_python" "$repair_945_solver_dir/mps_probe.py" \
    "$repair_945_output/945_repaired.npz" "$repair_945_output/945_tension_386.json" \
    386.025 386.082 --step .0005

"$repair_945_python" "$repair_945_solver_dir/replace_mps_bands.py" \
    "$repair_945_output/945_stage2.npz" "$repair_945_output/945_final.npz" \
    "$repair_945_output/945_tension_162.json" "$repair_945_output/945_tension_386.json" \
    --provenance "$repair_945_output/945_mps_provenance.json"
