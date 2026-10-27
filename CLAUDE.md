# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

prima.cpp is a fork of llama.cpp (late-2024 vintage) that runs one model across several home devices. Each device holds a contiguous window of layers, and the devices form a ring over ZeroMQ ("piped-ring" parallelism; a token may go around the ring several times). Weights are mmap'd and the next window is prefetched with `madvise`, so a device can hold more layers than its RAM. At startup rank 0 profiles every device and solves a linear program (the "Halda" scheduler, HiGHS) to decide how many layers each device runs and how many of them go on its GPU. On a single device it behaves like plain llama.cpp.

Paper: arXiv 2504.08791 (ICLR 2026). Its Appendix A.3 derives the Halda latency model.

## Build

The Makefile is the primary build. CMake exists but is not what the README uses.

```sh
make -j$(nproc)                          # CPU only
make GGML_CUDA=1 -j$(nproc)              # CUDA
make GGML_CUDA=1 USE_HIGHS=1 -j$(nproc)  # rank 0 must have USE_HIGHS=1 (Halda needs HiGHS)
make LLAMA_NO_METAL=1 -j$(nproc)         # macOS, very large models
make LLAMA_DEBUG=1 -j$(nproc)            # debug (slow)
```

Required system libraries: libzmq (linked as `-L/usr/local/lib -lzmq`), HiGHS ≥ 1.9 installed to `/usr/local` (rank 0 only), and `fio` (used by the disk-speed profiler).

Default targets are `llama-cli`, `llama-server`, `llama-speculative`, `llama-gguf-split` and `profile-tool`. Most upstream llama.cpp examples are commented out of `BUILD_TARGETS`.

Tests are the upstream llama.cpp unit tests. None of them cover the distributed path.

```sh
make tests                                         # build all TEST_TARGETS
make tests/test-backend-ops && ./tests/test-backend-ops
```

## Running

Single device: `./llama-cli -m model.gguf -c 1024 -p "..." -n 256 -ngl 30`

Distributed: start `llama-cli` on every device with the same `-m`. Each device also gets `--world N --rank R --master <rank0 ip> --next <next rank's ip>`, and usually `--prefetch`. Only rank 0 gets the prompt and generation arguments. Ports are `--data-port 9000` and `--signal-port 10000`. Rank 0 can run `llama-server` instead of `llama-cli`. Other options:

- `-lw "a,b,c,..."` (on rank 0): set each device's layer window by hand. This skips profiling and Halda. Workers then take `-ngl`.
- `--gpu-mem <GiB>`: cap the VRAM Halda may use on that device.
- `--keep-out-in-cuda`: run the output layer on the GPU. By default rank 0 computes the output layer on its CPU, which is expensive with a large vocab.
- `--force`: prefetch right after compute instead of only advising the OS.
- `profile-tool -m model.gguf`: run the device profiler by itself.

A device that Halda assigns no layers prints "No layer is assigned to me, exit". This is expected.

## Architecture (where the prima-specific code lives)

Almost everything prima adds is in three files. `ggml/` is mostly upstream.

- **`common/common.cpp`**
  - `llama_init_from_gpt_params` (~L1663) runs the distributed startup:
    1. Each rank calls `llama_profile_device`.
    2. Rank 0 collects the results with `llama_gather_device_info`.
    3. Rank 0 calls `assign_layers_and_select_devices`.
    4. Rank 0 sends the result with `llama_bcast_layer_setup`. Workers receive it with `llama_recv_layer_setup`.
    5. Every rank calls `llama_rebuild_topo`, which drops removed devices from the ring. Some of them may stay as forwarders.
  - Halda is `assign_layers_to_device` (~L860, compiled only with `USE_HIGHS`). It models per-token latency as a linear function of each device's layer count `w_m` and GPU layer count `n_m`, using the coefficients `a_m`, `b_m`, `c_m` (`xi`) and `kappa`. It sorts devices into memory cases M1–M4 and re-solves the LP until the case assignment stops changing. `assign_layers_and_select_devices` wraps it and removes slow devices.
- **`common/profiler.{h,cpp}`**: the `device_info` struct and the startup micro-benchmarks. These cover CPU and GPU FLOPS for each quantization type, RAM and VRAM read bandwidth, KV copy delay, disk read speed (through fio), and free memory. The benchmarks run once at startup. No network measurement exists.
- **`src/llama.cpp`** (very large; navigate with grep rather than reading it top to bottom)
  - Layer-window mapping: `n_layer_window[]`, around L3840–3880 and L7594.
  - ZMQ transport: `llama_send_meta` / `llama_recv_meta` carry the per-ubatch control data, and `llama_send_tensors` / `llama_recv_tensors` carry activations.
  - Socket setup and broadcasts: `llama_init_sockets`, `llama_bcast_*` and friends, around L20570–20960.
  - Decode: `llama_decode_internal` (~L18296) builds one ggml graph per contiguous layer window (`gf[]`). The subgraph loop works as follows:
    - It receives the input from the previous rank over `recv_socket` and computes.
    - It sends the result to the next rank. After the last layer, a worker sends to `master_socket`, and rank 0 runs the output layer.
    - Before moving on, it calls `manage_graph_tensors(..., POSIX_MADV_WILLNEED)` to prefetch the next window.
  - Pipelined prefill: when a batch has more than one ubatch, the loop runs window-major instead. An outer pass loop walks the layer windows and the ubatch loop sits inside it, so every rank pushes all ubatches through one window before starting the next and the ranks work on different ubatches at the same time. Every rank walks its windows in the same order, so tensors arrive in the order they are consumed. Rank 0 decides (`meta.pipelined`), KV slots for all ubatches are claimed before the pass loop, and `llama_count_layer_windows` disables it unless the windows tile the layers evenly. Single-token decode is unaffected.
- **`include/llama.h`**: the public API additions for all of the above (`llama_profile_device`, `llama_update_context_with_rankworld`, etc.).

## Profiling (current branch)

When `PRIMA_PROFILE=<csv path>` is set, `src/llama.cpp` uses the scheduler's eval callback to run each node separately and time it. The CSV gets one row per node, plus `ZMQ_RECV`, `ZMQ_SEND`, `D2H_COPY` and `SUBGRAPH` rows. Each rank writes its own CSV locally. The per-node sync slows inference down, so read the CSV for relative breakdowns, not absolute latency. While it is enabled, other `cb_eval` users are disabled.

## Known gaps between the Halda code and the paper

Check these before you treat a Halda prediction as meaningful:

- Several memory-read terms (`t_read_ram_*`) are commented out upstream. The `halda-mem-read` branch re-enables them.
- `xi` (the communication term) is fixed at 0.
- `kappa` uses layer FLOPs rather than the output-layer FLOPs.
- KV size uses `n_ctx`, not the KV length actually in use.
- Halda models only decode (TPOT). It does not model prefill.
- When the problem is infeasible on the first iteration, Halda hits a `GGML_ASSERT` instead of falling back.
