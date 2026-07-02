#!/bin/bash
set -euo pipefail
SCENES=(59 184 242 691 815 823 827 1312 1318 1459 1733 1784 1942 1996 2274 2765 3576 4992 5155 5447 5778 6732 6760 6784 6801)
#59 184 242 691 815 823 
FAILED=()
echo "Running all ${#SCENES[@]} scenes in one batch..."
bash run_perception.sh "${SCENES[@]}"
echo "=== DONE ==="
