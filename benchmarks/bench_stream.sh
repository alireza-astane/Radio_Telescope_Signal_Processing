#!/usr/bin/env bash
# Streaming correlator benchmark (csrc/stream): throughput vs number of
# antennas, a paced real-time run, and a bit-for-bit check against NumPy.
# Writes benchmarks/results/stream.csv.  Usage: benchmarks/bench_stream.sh [--quick]
set -euo pipefail
cd "$(dirname "$0")/.."
RX=csrc/stream/rx_sim
CORRD=csrc/stream/corrd
OUT=benchmarks/results/stream.csv
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PORT=${PORT:-5300}
FRAME=4096
NCHAN=64
ANTS="2 4 8 16 32"
FRAMES=2000
if [[ "${1:-}" == "--quick" ]]; then ANTS="2 4"; FRAMES=200; OUT=benchmarks/results/stream_quick.csv; fi

[[ -x $RX && -x $CORRD ]] || make -C csrc stream >/dev/null

run() { # port n_ant frames rate [extra rx_sim args]; prints corrd CSV fields, then "overruns,max_lag_ms"
    local port=$1 n=$2 frames=$3 rate=$4; shift 4
    $CORRD --n-ant "$n" --port "$port" --nchan $NCHAN --frame $FRAME --integrate 16 "${CORRD_ARGS[@]}" \
        >"$TMP/corrd.log" 2>"$TMP/corrd.err" &
    local pid=$!
    sleep 0.2
    $RX --port "$port" --n-ant "$n" --frame $FRAME --frames "$frames" --rate "$rate" "$@" >"$TMP/rx.log"
    wait $pid
    grep '^CSV' "$TMP/corrd.log" | cut -d, -f2-
    if grep -q 'max lag' "$TMP/rx.log"; then
        sed -n 's/.*max lag \([0-9.]*\) ms, \([0-9]*\) overruns.*/\2,\1/p' "$TMP/rx.log"
    else
        echo "0,0"
    fi
}

echo "mode,n_ant,nchan,frame,frames,lost,elapsed_s,t_fstep_s,t_xstep_s,msps_per_ant,overruns,max_lag_ms,rate_msps" >"$OUT"

CORRD_ARGS=()
for n in $ANTS; do
    PORT=$((PORT + 1)); mapfile -t r < <(run $PORT "$n" $FRAMES 0)
    echo "unlimited,${r[0]},${r[1]},0" >>"$OUT"
    echo "unlimited N=$n -> $(echo "${r[0]}" | cut -d, -f9) MS/s per antenna"
done

# Paced at the nominal 2.4 MS/s per receiver for ~2 s.  An overrun (lag beyond a
# 32-frame / 55 ms receiver buffer) would mean lost data in a real deployment.
for n in 8 16; do
    [[ " $ANTS " == *" $n "* ]] || continue
    PORT=$((PORT + 1)); mapfile -t r < <(run $PORT "$n" 1200 2.4)
    echo "paced,${r[0]},${r[1]},2.4" >>"$OUT"
    echo "paced 2.4 MS/s N=$n -> overruns,max_lag_ms: ${r[1]}"
done

# Correctness: stream a simulated observation through the daemon and compare
# its visibility dumps with the offline NumPy correlator.
python3 benchmarks/stream_tools.py make-input "$TMP/in.bin" --n-ant 4 --samples 262144
CORRD_ARGS=(--out "$TMP/vis.bin")
PORT=$((PORT + 1)); run $PORT 4 64 0 --input "$TMP/in.bin" >/dev/null
python3 benchmarks/stream_tools.py verify "$TMP/in.bin" "$TMP/vis.bin" --frame $FRAME --nchan $NCHAN
echo "wrote $OUT"
