# CPU speed probe

Same 10 spoken-form sentences (`sets/dev_numbers10.jsonl`, field `spoken`) for every system, after one warm-up sentence; latest run per (system, threads, OpenMP wait policy). RTF = wall seconds / audio seconds; CPU-RTF = process CPU seconds / audio seconds. The machine is shared with other jobs: compare load averages before reading much into small differences. `PASSIVE` (the harness default) avoids the large slowdown seen when the CPU is oversubscribed (EMA Lightning RTF 1.37 with the default vs 0.26 with PASSIVE at load ~9); at the light load of these probes the OpenMP `default` (spin, then sleep) was within about 20 %. pocket-tts runs its decoder in a second Python thread, so its CPU-RTF exceeds its RTF even at 1 torch thread.

| system | threads | OMP wait policy | RTF | CPU-RTF | audio s | load avg (start-end) | when (UTC) |
|---|---:|---|---:|---:|---:|---|---|
| ema-lightning | 2 | PASSIVE | 0.177 | 0.195 | 39.8 | 1.23-1.19 | 2026-10-07T06:14 |
| ema-lightning | 2 | default | 0.155 | 0.285 | 39.8 | 1.27-1.31 | 2026-10-07T06:19 |
| mms-tts-tur | 2 | PASSIVE | 0.327 | 0.532 | 56.2 | 1.26-1.39 | 2026-10-07T06:15 |
| mms-tts-tur | 2 | default | 0.308 | 0.572 | 56.2 | 1.71-1.73 | 2026-10-07T06:21 |
| piper-tr-dfki | 2 | PASSIVE | 0.068 | 0.109 | 44.6 | 1.37-1.26 | 2026-10-07T06:15 |
| piper-tr-dfki | 2 | default | 0.068 | 0.108 | 44.6 | 2.29-1.86 | 2026-10-07T06:20 |
| pocket-tts-turkish | 1 | PASSIVE | 0.562 | 0.812 | 44.7 | 1.65-1.55 | 2026-10-07T06:19 |
| pocket-tts-turkish | 1 | default | 0.532 | 0.748 | 44.7 | 2.07-1.86 | 2026-10-07T06:24 |
| pocket-tts-turkish | 2 | PASSIVE | 0.456 | 0.973 | 44.7 | 1.19-1.4 | 2026-10-07T06:14 |
| pocket-tts-turkish | 2 | default | 0.527 | 1.765 | 44.7 | 1.31-2.29 | 2026-10-07T06:20 |
| qwen3-tts-tiny-tr | 2 | PASSIVE | 3.327 | 4.650 | 51.6 | 1.36-1.65 | 2026-10-07T06:19 |
| qwen3-tts-tiny-tr | 2 | default | 2.692 | 5.070 | 51.6 | 1.75-2.07 | 2026-10-07T06:23 |
