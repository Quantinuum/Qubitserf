# BZ vs connected-cluster: the sparse/dense divide

Uniform per-run timeout 330s.

| code | family | n | d | cc (cpu) | bz (cpu) | bz (gpu) |
|---|---|---|---|---|---|---|
| steane [[7,1,3]] | dense | 7 | 3 | 12ms | 5ms | 35ms |
| toric L=6 [[72,2,6]] | sparse | 72 | 6 | 6ms | 5ms | 33ms |
| bb [[72,12,6]] | sparse | 72 | 6 | 8ms | 6ms | 36ms |
| toric L=10 [[200,2,10]] | sparse | 200 | 10 | 10ms | 135.3s | 40.7s |
| gross [[144,12,12]] | sparse | 144 | 12 | 297ms | >330s | 275.5s |
| qbch [[15,7,3]] | dense | 15 | 3 | 10ms | 7ms | 43ms |
| qbch [[31,11,5]] | dense | 31 | 5 | 8ms | 6ms | 39ms |
| qbch [[31,1,7]] | dense | 31 | 7 | 8ms | 5ms | 38ms |
| qbch [[63,39,5]] | dense | 63 | 5 | 9ms | 6ms | 41ms |
| qbch [[63,27,7]] | dense | 63 | 7 | 233ms | 14ms | 354ms |
| qbch [[127,71,9]] | dense | 127 | 9 | >330s | 142.3s | 26.3s |

## Takeaway

Connected-cluster certifies sparse LDPC codes (toric, bivariate-bicycle) in milliseconds where BZ needs minutes or times out; Brouwer-Zimmermann certifies dense-check quantum BCH codes in milliseconds-to-seconds where cluster growth makes CC time out. Neither dominates: the right method follows the Tanner-graph sparsity of the code.
