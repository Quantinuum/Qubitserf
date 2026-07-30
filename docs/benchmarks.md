# Benchmarks

Computed on an M4 MacBook.

## Brouwer–Zimmermann: CPU vs GPU

### CSS quantum codes

| code | n | k | d(cpu) | d(gpu) | t_cpu | t_gpu |
|---|---|---|---|---|---|---|
| toric L=4 | 32 | 2 | 4 | 4 | 489us | 296us |
| toric L=5 | 50 | 2 | 5 | 5 | 1.3ms | 942us |
| toric L=6 | 72 | 2 | 6 | 6 | 2.4ms | 2.0ms |
| toric L=7 | 98 | 2 | 7 | 7 | 378.1ms | 320.8ms |
| toric L=8 | 128 | | 8 | 8 | 2.12s | 971.8ms |
| surface L=4 | 25 | 1 | 4 | 4 | 484us | 297us |
| surface L=5 | 41 | 1 | 5 | 5 | 885us | 652us |
| surface L=6 | 61 | 1 | 6 | 6 | 3.2ms | 3.1ms |
| surface L=7 | 85 | 1 | 7 | 7 | 142.8ms | 439.5ms |
| hgp(rep6,rep6) | 61 | 1 | 6 | 6 | 3.2ms | 3.1ms |
| hgp(ham3,ham3) | 58 | 16 | 3 | 3 | 725us | 521us |

### Classical codes

| code | n | d(cpu) | d(gpu) | t_cpu | t_gpu |
|---|---|---|---|---|---|
| hamming r=3 | 7 | 3 | 3 | 197us | 108us |
| hamming r=4 | 15 | 3 | 3 | 266us | 129us |
| repetition n=6 | 6 | 6 | 6 | 228us | 98us |
| repetition n=8 | 8 | 8 | 8 | 127us | 120us |
| rand_ldpc(12,18,3) | 18 | 2 | 2 | 320us | 297us |

## Connected cluster vs BZ

| code | n | d | cc | bz |
|---|---|---|---|---|
| steane [[7,1,3]] | 7 | 3 | d=3 (1.4ms) | d=3 (245us) |
| shor [[9,1,3]] | 9 | 3 | d=3 (824us) | d=3 (274us) |
| toric L=6 [[72,2,6]] | 72 | 6 | d=6 (1.1ms) | d=6 (2.1ms) |
| surface L=6 [[61,1,6]] | 61 | 6 | d=6 (1.0ms) | d=6 (3.1ms) |
| bb [[72,12,6]] | 72 | 6 | d=6 (1.2ms) | [6,6] (3.2ms) |
| toric L=9 [[162,2,9]] | 162 | 9 | d=9 (1.9ms) | [7,9] (6.02s) |
| toric L=10 [[200,2,10]] | 200 | 10 | d=10 (2.5ms) | [8,10] (27.51s) |
| **gross [[144,12,12]]** | 144 | 12 | **d=12 (377.6ms)** | [8,12] (3.87s) |
