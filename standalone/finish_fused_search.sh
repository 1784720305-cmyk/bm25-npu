#!/usr/bin/env bash
# Run only AFTER the currently running frozen small acceptance has completed.
# No kernel generation/build/install/retry here; preserves that job and its parameters.
set -eo pipefail
cd /test/w50066029/bm25-npu
device="${1:-5}"
acceptance="${2:-results/fused_chain_small_s2_ubfix1}"
result="${3:-results/fused_search_256k_q8_close1}"
python3 - "$acceptance" "$result" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'report.json'
if not p.is_file():raise SystemExit('STOP: current device acceptance is missing; do not restart it')
if json.loads(p.read_text(encoding='utf-8')).get('status')!='fused_chain_actual_debug_modes_and_all_rankings_passed':
    raise SystemExit('STOP: current acceptance is running/failed/not accepted; do not benchmark or restart')
if Path(sys.argv[2]).exists():raise SystemExit('STOP: closure output exists; do not overwrite/restart')
PY
source /usr/local/Ascend/cann-9.2.0/set_env.sh
python3 verify_package.py
cmake -S npu_prototype/fused_search_host -B results/fused_search_host_build -DTOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0 -DBM25_VENDOR_DIR=/test/w50066029/bm25-npu/results/custom_opp/vendors 2>&1 | tee results/fused_search_configure1.log
cmake --build results/fused_search_host_build -j 2>&1 | tee results/fused_search_build1.log
# Capability/old build evidence, no kernel recompile or --cce-res-usage flag changes.
python3 standalone/inspect_fused_hardware.py --device "$device" --output "${result}_hardware.json" 2>&1 | tee "${result}_hardware.log"
python3 standalone/close_fused_search.py --fused-acceptance "$acceptance" --device "$device" --output "$result" 2>&1 | tee "${result}.log"
