# qminweight connected-cluster benchmark

`cc` = qminweight connected cluster; `bz` = qminweight Brouwer-Zimmermann (capped on hard codes, shown as the rigorous `[lower,upper]` bracket); reference = `codeDistance` package.

| code | n | d | qminweight cc | qminweight bz | ref BZDistMW | ref connectedClusterMW |
|---|---|---|---|---|---|---|
| steane [[7,1,3]] | 7 | 3 | d=3 (1.4ms) | d=3 (245us) | d=3 (3.3ms) | d=3 (816us) |
| shor [[9,1,3]] | 9 | 3 | d=3 (824us) | d=3 (274us) | d=3 (1.4ms) | d=0 (741us) |
| toric L=6 [[72,2,6]] | 72 | 6 | d=6 (1.1ms) | d=6 (2.1ms) | d=6 (239.6ms) | d=6 (5.3ms) |
| surface L=6 [[61,1,6]] | 61 | 6 | d=6 (1.0ms) | d=6 (3.1ms) | d=6 (140.7ms) | d=6 (3.8ms) |
| bb [[72,12,6]] | 72 | 6 | d=6 (1.2ms) | [6,6] capped (3.2ms) | d=6 (1.10s) | d=6 (32.3ms) |
| toric L=9 [[162,2,9]] | 162 | 9 | d=9 (1.9ms) | [7,9] capped (6.02s) | >30s (timeout) | d=9 (157.5ms) |
| toric L=10 [[200,2,10]] | 200 | 10 | d=10 (2.5ms) | [8,10] capped (27.51s) | >30s (timeout) | d=10 (522.5ms) |
| gross [[144,12,12]] | 144 | 12 | d=12 (377.6ms) | [8,12] capped (3.87s) | >30s (timeout) | >30s (timeout) |

## Takeaway

On sparse codes whose BZ lower bound is weak (bivariate-bicycle, large toric), qminweight's connected cluster certifies the exact distance in well under a second, while Brouwer-Zimmermann (qminweight's and the reference's) cannot prove it in tractable time. On small codes all methods agree.

qminweight's CC is the *same algorithm* as the reference's `connectedClusterMW`, but a compiled, seed-parallel implementation: it is tens to hundreds of times faster (e.g. toric L=10: 2.5 ms vs 523 ms, ~209×), and it certifies the gross code `[[144,12,12]]` in ~0.4 s where the reference's own connected cluster times out (>30 s). (qminweight's CC also returns the correct d=3 for the Shor code, where the reference's `connectedClusterMW` reports a spurious d=0.)
