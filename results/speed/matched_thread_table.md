# WS8.1 — matched-thread wall-clock and peak memory

**Wall-clock definition (identical for every tool):** process start to results file written and process exited, including interpreter start, imports, model/database load, and output write. Instrument: `/usr/bin/time -v`. No `conda run` wrapper. Uncompressed `.fasta` read from a local ext4 filesystem (`/dev/sdb`), warm page cache unless stated.

**Hardware:** 48-core host (2 sockets), 881 GiB RAM, ext4 on /dev/sdb. CPU-only ONNX / CPU-only PyTorch; no GPU was used by any tool.

**Every cell is median [min–max] over n repeats.** The 1-minute load average was recorded immediately before and after every run and is in `matched_thread_runs.tsv`; the campaign driver waits for the load average to fall below 1.0 before starting each timed run.

---

## Primary matched factorial — set_E_100 (**denominator: 100 genomes, 0.478 Gbp**, a seeded 10% subsample of Set E)

| Tool | 1 thread | 8 threads | 16 threads | 32 threads | Peak RSS (GB) |
|---|---|---|---|---|---|
| MAGICC v0.3.0 (V5) | **5.9 s**<br><sub>5.9 s–6.0 s, n=3</sub> | **2.2 s**<br><sub>2.1 s–2.2 s, n=3</sub> | **2.1 s**<br><sub>2.1 s–2.1 s, n=3</sub> | **2.2 s**<br><sub>2.1 s–13.2 s, n=3</sub> | 0.39 |
| CheckM2 1.0.1 | **2 h 11 m**<br><sub>2 h 11 m–2 h 11 m, n=3</sub> | **20 m 08.7 s**<br><sub>19 m 55.2 s–20 m 33.5 s, n=3</sub> | **11 m 27.9 s**<br><sub>11 m 19.7 s–12 m 03.8 s, n=3</sub> | **9 m 19.7 s**<br><sub>9 m 13.0 s–9 m 23.6 s, n=3</sub> | 10.66 |
| CoCoPyE 0.5.0 | **23 m 59.1 s**<br><sub>23 m 37.3 s–26 m 32.5 s, n=3</sub> | **4 m 59.1 s**<br><sub>4 m 57.0 s–5 m 25.1 s, n=3</sub> | **3 m 36.1 s**<br><sub>3 m 33.4 s–3 m 36.4 s, n=3</sub> | **5 m 53.9 s**<br><sub>5 m 39.4 s–8 m 15.1 s, n=3</sub> | 15.89 |
| DeepCheck (inference only) | **55.5 s**<br><sub>55.3 s–55.6 s, n=3</sub> | **9.8 s**<br><sub>9.7 s–9.8 s, n=3</sub> | **6.4 s**<br><sub>6.4 s–6.5 s, n=3</sub> | **5.0 s**<br><sub>5.0 s–5.0 s, n=3</sub> | 0.71 |


## Full-scale anchor — Set E (**denominator: 1000 genomes, 4.807 Gbp**, the historical Table S4 input)

| Tool | 1 thread | 8 threads | 16 threads | 32 threads | Peak RSS (GB) |
|---|---|---|---|---|---|
| MAGICC v0.3.0 (V5) | **49.0 s**<br><sub>48.6 s–49.3 s, n=6</sub> | **10.3 s**<br><sub>10.3 s–10.3 s, n=3</sub> | **6.9 s**<br><sub>6.9 s–7.3 s, n=3</sub> | **6.0 s**<br><sub>5.8 s–39.1 s, n=3</sub> | 0.65 |
| CheckM2 1.0.1 | not run | not run | not run | **1 h 10 m**<br><sub>1 h 10 m–1 h 10 m, n=1</sub> | 18.88 |
| CoCoPyE 0.5.0 | not run | not run | not run | **52 m 57.5 s**<br><sub>52 m 57.5 s–52 m 57.5 s, n=1</sub> | 15.90 |
| DeepCheck (inference only) | **8 m 54.6 s**<br><sub>8 m 54.4 s–8 m 57.3 s, n=3</sub> | **1 m 18.1 s**<br><sub>1 m 18.0 s–1 m 18.7 s, n=3</sub> | **43.5 s**<br><sub>42.7 s–43.8 s, n=3</sub> | **29.5 s**<br><sub>29.2 s–39.8 s, n=3</sub> | 1.29 |

