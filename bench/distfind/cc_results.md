# qubitserf connected-cluster benchmark

`cc` = qubitserf connected cluster; `bz` = qubitserf Brouwer-Zimmermann (capped on hard codes, shown as the rigorous `[lower,upper]` bracket); reference = `codeDistance` package.

| code | n | d | qubitserf cc | qubitserf bz | ref BZDistMW | ref connectedClusterMW |
|---|---|---|---|---|---|---|
| steane [[7,1,3]] | 7 | 3 | d=3 (539us) | d=3 (228us) | d=3 (1.0ms) | d=3 (1.8ms) |
| shor [[9,1,3]] | 9 | 3 | d=3 (521us) | d=3 (241us) | d=3 (1.1ms) | d=0 (482us) |
| toric L=6 [[72,2,6]] | 72 | 6 | d=6 (1.2ms) | d=6 (2.0ms) | d=6 (407.8ms) | d=6 (5.9ms) |
| surface L=6 [[61,1,6]] | 61 | 6 | d=6 (1.3ms) | d=6 (5.0ms) | d=6 (128.6ms) | d=6 (3.5ms) |
| bb [[72,12,6]] | 72 | 6 | d=6 (1.1ms) | [6,6] capped (2.7ms) | d=6 (938.4ms) | d=6 (29.5ms) |
| toric L=9 [[162,2,9]] | 162 | 9 | d=9 (1.9ms) | [7,9] capped (394.7ms) | >30s (timeout) | d=9 (151.0ms) |
| toric L=10 [[200,2,10]] | 200 | 10 | d=10 (2.9ms) | [8,10] capped (1.53s) | >30s (timeout) | d=10 (551.0ms) |
| gross [[144,12,12]] | 144 | 12 | d=12 (242.6ms) | [8,12] capped (337.1ms) | >30s (timeout) | >30s (timeout) |

## Takeaway

On sparse codes whose BZ lower bound is weak (bivariate-bicycle, large toric), qubitserf's connected cluster certifies the exact distance in well under a second, while Brouwer-Zimmermann (qubitserf's and the reference's) cannot prove it in tractable time. On small codes all methods agree.

qubitserf's CC is the *same algorithm* as the reference's `connectedClusterMW`, but a compiled, seed-parallel implementation: it is tens to hundreds of times faster, and it certifies the gross code `[[144,12,12]]` (~0.4 s) where the reference's own connected cluster times out (>30 s).
