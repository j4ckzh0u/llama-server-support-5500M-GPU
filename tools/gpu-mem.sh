#!/bin/sh
# gpu-mem: what an Intel Mac's AMD card holds right now (macOS has no nvidia-smi; this is the nearest thing).
#
# gpu-mem the card's counters, once
# gpu-mem -w [SEC] refresh every SEC seconds (default 2)
# gpu-mem -s [LOG] the counters plus llama-server's own buffer breakdown from LOG
# (default /tmp/llama-server.log; start the server with -lv4 to get it)
#
# Each GPU's driver counters come from the IORegistry (one PerformanceStatistics dictionary per accelerator):
# VRAM inUseVidMemoryBytes the card's own memory: weights copied into VRAM, KV cache, compute buffers
# SYS inUseSysMemoryBytes system-RAM buffers the GPU reads (a shared/mapped model, IOSurfaces)
# GART gartUsedBytes/gartSizeBytes GPU address space in use (VRAM + system memory mapped into the GPU)
# macOS exposes no per-process split of the card's memory; llama.cpp's -lv4 lines give that per buffer.
set -u

show() {
 tmp=/tmp/.gpu-mem.$$
 ioreg -r -c IOAccelerator -l > "$tmp"
 printf 'GPU memory (driver counters, IORegistry)\n'
 system_profiler SPDisplaysDataType |
 awk -F': ' '/Chipset Model:/{printf " present: %s\n", $2}' | head -2
 printf ' %-40s %9s %9s %9s %9s\n' GPU VRAM_MB SYS_MB GART_MB GART_TOT
 awk '
 /^\+-o / {
 gpu = $2
 sub(/GraphicsAccelerator$/, "", gpu)
 sub(/^IntelAccelerator$/, "Intel iGPU", gpu)
 sub(/^AMDRadeonX6000_AMDNavi14/, "AMD Navi14", gpu)
 next
 }
 /"PerformanceStatistics" = / {
 rest = $0
 while (match(rest, /"inUseVidMemoryBytes"=[0-9]+/)) { s = substr(rest, RSTART, RLENGTH); sub(/.*=/, "", s); if (s +0 > vid[gpu]) vid[gpu] = s +0; rest = substr(rest, RSTART + RLENGTH) }
 rest = $0
 while (match(rest, /"inUseSysMemoryBytes"=[0-9]+/)) { s = substr(rest, RSTART, RLENGTH); sub(/.*=/, "", s); if (s +0 > sysm[gpu]) sysm[gpu] = s +0; rest = substr(rest, RSTART + RLENGTH) }
 rest = $0
 while (match(rest, /"gartUsedBytes"=[0-9]+/)) { s = substr(rest, RSTART, RLENGTH); sub(/.*=/, "", s); if (s +0 > gart[gpu]) gart[gpu] = s +0; rest = substr(rest, RSTART + RLENGTH) }
 rest = $0
 while (match(rest, /"gartSizeBytes"=[0-9]+/)) { s = substr(rest, RSTART, RLENGTH); sub(/.*=/, "", s); if (s +0 > gsize[gpu]) gsize[gpu] = s +0; rest = substr(rest, RSTART + RLENGTH) }
 }
 END {
 n =0
 for (g in gsize) {
 printf " %-40s %9.1f %9.1f %9.1f %9.1f\n", g, vid[g] /1e6, sysm[g] /1e6, gart[g] /1e6, gsize[g] /1e6
 vt += vid[g] /1e6
 st += sysm[g] /1e6
 n++
 }
 printf " %-40s %9.1f %9.1f\n", "total (" n " GPUs)", vt, st
 }' "$tmp"
 rm -f "$tmp"
}

breakdown() {
 log="$1"
 if [ -f "$log" ]; then
 printf '\nserver buffers (%s):\n' "$log"
 grep -E 'load_tensors:.*buffer size|llama_kv_cache:.*buffer size|llama_memory_recurrent:.*buffer size|sched_reserve:.*buffer size' "$log" |
 sed -E 's/^[0-9.]+ [A-Za-z]+ +//' | sort -u | sed 's/^/ /'
 else
 printf '\n(no log at %s: start llama-server with -lv4 so it prints each buffer)\n' "$log"
 fi
 pid=$(pgrep -f 'llama-server' | head -1)
 if [ -n "${pid:-}" ]; then
 printf '\nllama-server pid %s, process footprint (all memory, not only the GPU):\n' "$pid"
 footprint -p "$pid" | grep -E 'phys_footprint' | sed 's/^/ /'
 fi
}

case "${1:-}" in
-w|--watch)
 every="${2:-2}"
 case "$every" in ''|*[!0-9]*) every=2 ;; esac
 while :; do
 clear
 date '+%H:%M:%S'
 show
 sleep "$every"
 done
 ;;
-s|--server)
 show
 breakdown "${2:-/tmp/llama-server.log}"
 ;;
*)
 show
 ;;
esac
