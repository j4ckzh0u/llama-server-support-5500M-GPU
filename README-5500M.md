# llama.cpp for an Intel Mac with a Radeon Pro 5500M

This is llama.cpp at commit `3cf03257` (the commit Strata pins) with three fixes an Intel Mac's discrete AMD card needs.
Every number below was measured on a MacBook Pro 16,1 (2019), Intel Core i9-9980HK, 64 GB RAM, AMD Radeon Pro 5500M 8 GB,
macOS 15.7.5.

## The three problems, and what changed

| What you see | Why | Fix in this tree |
| --- | --- | --- |
| `GGML_ASSERT(buf_src) failed` in `ggml_metal_buffer_set_tensor` | a private Metal buffer copies from a host pointer that is not page-aligned; `newBufferWithBytesNoCopy` then returns nil | unaligned data is staged in a page-aligned copy (`ggml/src/ggml-metal/ggml-metal-device.m`) |
| Correct on the CPU, garbage (repeated BOS) on the GPU | `MTLDispatchTypeConcurrent` command buffers race on this card: the memory barriers do not order the dispatches | `GGML_METAL_CONCURRENCY_DISABLE=1` (an existing switch) |
| 1.55 tokens/s, the GPU barely used | `ggml_metal_buffer_map` wraps the mmapped GGUF as `MTLResourceStorageModeShared`, so the weights stay in system RAM and every decode step reads them over PCIe (measured 1.4 to 1.85 GB/s) | `GGML_METAL_WEIGHTS_VRAM=1` copies them into a private (VRAM) buffer at load; a model over `maxBufferLength` (3.76 GB here) is copied in chunks |
| a private mapped buffer asserts in the shared interface | `ggml_backend_metal_device_buffer_mapped` always used the shared interface | it picks the interface that matches the buffer's storage (`ggml/src/ggml-metal/ggml-metal.cpp`) |

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
ninja -C build llama-server llama-cli
```

The first start compiles the Metal libraries on the CPU (about 65 s here); later starts reuse the driver's cache.

## Run

```sh
GGML_METAL_WEIGHTS_VRAM=1 GGML_METAL_SHARED_BUFFERS_DISABLE=1 GGML_METAL_CONCURRENCY_DISABLE=1 \
 ./build/bin/llama-server -m model.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8080
```

`GGML_METAL_SHARED_BUFFERS_DISABLE=1` is optional: without it the KV and compute buffers stay in system RAM. With it they
stay in VRAM too, which measured 9.6 against 7.4 tokens/s at a 1.4K-token context.

## Watching VRAM (`tools/gpu-mem.sh`)

macOS has no `nvidia-smi` and no per-process VRAM API for a discrete card. Two sources exist, and `tools/gpu-mem.sh`
reads both: the driver's own counters from the IORegistry, and llama.cpp's per-buffer accounting from `-lv4`.

```sh
tools/gpu-mem.sh # this moment
tools/gpu-mem.sh -w 2 # refresh every 2 s
tools/gpu-mem.sh -s /tmp/llama-server.log # counters + the server's buffers
```

With Qwen3.5-4B, `-ngl 99 -c 4096`, weights and KV/compute in VRAM (`-lv4` log):

```
GPU memory (driver counters, IORegistry)
 present: Intel UHD Graphics 630
 present: AMD Radeon Pro 5500M
 GPU                                       VRAM_MB   SYS_MB  GART_MB GART_TOT
 Intel iGPU                                    0.0    674.4   1559.8   7516.2
 AMD Navi14                                 3797.1    183.6    183.6  23622.3
 total (2 GPUs)                             3797.1    858.0                  

server buffers (/tmp/llama-server.log):
 llama_kv_cache: MTL0_Private KV buffer size = 128.00 MiB
 llama_memory_recurrent: MTL0_Private RS buffer size = 201.00 MiB
 load_tensors: CPU_Mapped model buffer size = 497.31 MiB
 load_tensors: MTL0_Mapped model buffer size = 2603.50 MiB
 sched_reserve: CPU compute buffer size = 18.02 MiB
 sched_reserve: MTL0_Private compute buffer size = 182.02 MiB

llama-server pid 68755, process footprint (all memory, not only the GPU):
 phys_footprint: 100 MB
 phys_footprint_peak: 2675 MB
```

How to read it:

| Counter | What it is | This run |
| --- | --- | --- |
| `VRAM_MB` (`inUseVidMemoryBytes`) | the card's own memory: weights copied into VRAM, KV cache, compute buffers | 3,797 MB |
| `SYS_MB` (`inUseSysMemoryBytes`) | system-RAM buffers the GPU reads (a shared/mapped model, IOSurfaces) | 184 MB |
| `GART_MB` / `GART_TOT` | GPU address space in use / its size | 184 / 23,622 MB |

The breakdown adds up: the weights (2,603.50 MiB mapped, `-lv4` says `MTL0_Mapped`) plus the KV cache (128), the
recurrent state of the hybrid layers (201), the compute buffers (182) and some allocation overhead come to the
3,797 MB the driver reports. Without `GGML_METAL_WEIGHTS_VRAM=1` the same model shows about 500 MB of VRAM and the
weights move to `SYS_MB` (system memory read over PCIe) — which is exactly why it ran at 1.55 tokens/s.

Notes:

- `phys_footprint` is the whole process, not the GPU: 100 MB here, because the weights are in VRAM and the mmapped
 file pages are clean page cache.
- On this Mac the IORegistry lists two accelerators (the AMD card and the Intel iGPU); the script prints one row each.
- The counters are the driver's, so they include whatever else uses the card (the display, other apps).

## Measured

Qwen3.5-4B Q4_K_M (2.74 GB), greedy, 8K context, `-ngl 99`:

| Configuration | Decode | Prompt |
| --- | ---: | ---: |
| as shipped (weights mapped shared) | 1.55 tok/s | 2.1 tok/s |
| CPU only (`-ngl 0`, the i9) | 5.93 tok/s | 19.0 tok/s |
| weights in VRAM, KV in RAM | 15.4 tok/s | 34.6 tok/s |
| weights and KV/compute in VRAM | 23.1 tok/s | 35.9 tok/s |
| `llama-server`, same flags | 26.3 tok/s | 36.5 tok/s |

A model has to fit the 8.57 GB working set for this speed: a 4B or 7B Q4 does, a 16 GB model does not.

`--spec-type draft-mtp` needs a model with MTP/NextN tensors, and Qwen3.5-4B has none. The ngram self-speculative types
measured 22.4 (`ngram-simple`) to 24.0 (`ngram-mod`) tokens/s against 22.6 without them on a repeating-line prompt at
temperature 0; `ngram-cache` was slower (14.4). So speculative decoding is not the lever on this card.
