# llama.cpp for an Intel Mac with a Radeon Pro5500M

This is llama.cpp at commit `3cf03257` (the commit Strata pins) with three fixes an Intel Mac's discrete AMD card needs.
Every number below was measured on a MacBook Pro16,1 (2019), Intel Core i9-9980HK,64 GB RAM, AMD Radeon Pro5500M8 GB,
macOS15.7.5.

## The three problems, and what changed

| What you see | Why | Fix in this tree |
| --- | --- | --- |
| `GGML_ASSERT(buf_src) failed` in `ggml_metal_buffer_set_tensor` | a private Metal buffer copies from a host pointer that is not page-aligned; `newBufferWithBytesNoCopy` then returns nil | unaligned data is staged in a page-aligned copy (`ggml/src/ggml-metal/ggml-metal-device.m`) |
| Correct on the CPU, garbage (repeated BOS) on the GPU | `MTLDispatchTypeConcurrent` command buffers race on this card: the memory barriers do not order the dispatches | `GGML_METAL_CONCURRENCY_DISABLE=1` (an existing switch) |
|1.55 tokens/s, the GPU barely used | `ggml_metal_buffer_map` wraps the mmapped GGUF as `MTLResourceStorageModeShared`, so the weights stay in system RAM and every decode step reads them over PCIe (measured1.4 to1.85 GB/s) | `GGML_METAL_WEIGHTS_VRAM=1` copies them into a private (VRAM) buffer at load; a model over `maxBufferLength` (3.76 GB here) is copied in chunks |
| a private mapped buffer asserts in the shared interface | `ggml_backend_metal_device_buffer_mapped` always used the shared interface | it picks the interface that matches the buffer's storage (`ggml/src/ggml-metal/ggml-metal.cpp`) |

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
ninja -C build llama-server llama-cli
```

The first start compiles the Metal libraries on the CPU (about65 s here); later starts reuse the driver's cache.

## Run

```sh
GGML_METAL_WEIGHTS_VRAM=1 GGML_METAL_SHARED_BUFFERS_DISABLE=1 GGML_METAL_CONCURRENCY_DISABLE=1 \
 ./build/bin/llama-server -m model.gguf -ngl99 -c8192 --host127.0.0.1 --port8080
```

`GGML_METAL_SHARED_BUFFERS_DISABLE=1` is optional: without it the KV and compute buffers stay in system RAM. With it they
stay in VRAM too, which measured9.6 against7.4 tokens/s at a1.4K-token context.

## Measured

Qwen3.5-4B Q4_K_M (2.74 GB), greedy,8K context, `-ngl99`:

| Configuration | Decode | Prompt |
| --- | ---: | ---: |
| as shipped (weights mapped shared) |1.55 tok/s |2.1 tok/s |
| CPU only (`-ngl0`, the i9) |5.93 tok/s |19.0 tok/s |
| weights in VRAM, KV in RAM |15.4 tok/s |34.6 tok/s |
| weights and KV/compute in VRAM |23.1 tok/s |35.9 tok/s |
| `llama-server`, same flags |26.3 tok/s |36.5 tok/s |

A model has to fit the8.57 GB working set for this speed: a4B or7B Q4 does, a16 GB model does not.

`--spec-type draft-mtp` needs a model with MTP/NextN tensors, and Qwen3.5-4B has none. The ngram self-speculative types
measured22.4 (`ngram-simple`) to24.0 (`ngram-mod`) tokens/s against22.6 without them on a repeating-line prompt at
temperature0; `ngram-cache` was slower (14.4). So speculative decoding is not the lever on this card.
