/* codeaut -- flat extern "C" ABI for the native backends (loaded from Python via ctypes).
 *
 * Two engines:
 *   - Leon code-automorphism backend  (qaut_leon_*)  -- defined in src/leon.cpp
 *   - Brouwer-Zimmermann low-weight enumerator (codeaut_bz_*) -- defined in src/bz.cpp,
 *     dispatching to a CPU / CUDA / Metal backend (src/backend_cpu.cpp, src/cuda, src/metal).
 *
 * No external dependency: each function is a C symbol called over ctypes.
 */
#ifndef CODEAUT_CAPI_H
#define CODEAUT_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ Leon engine */
/* Run Leon's algorithm on the (m x n, row-major uint8) generator matrix G.
 * Returns an opaque handle (NULL on allocation failure); free with qaut_leon_free. */
void*   qaut_leon_run(const uint8_t* G, int32_t m, int32_t n,
                      int32_t max_dim, int32_t use_invariant);
/* Extended selector: 0 = legacy ascending minimum-weight prefix; 1 = smallest complete
 * spanning weight-congruence class; 2 = cost-aware hybrid (probe congruences only for a large
 * prefix, and use one only when it has fewer word vertices); 3 = support-minimal/cocircuit
 * filtering of the legacy prefix. max_modulus <= 0 searches n+1. */
void*   qaut_leon_run_ex(const uint8_t* G, int32_t m, int32_t n,
                         int32_t max_dim, int32_t use_invariant,
                         int32_t selector, int32_t max_modulus);
int32_t qaut_leon_ok(void* h);             /* 1 if dim(C) <= max_dim (result valid)         */
int32_t qaut_leon_dim(void* h);            /* dim(C)                                         */
int32_t qaut_leon_n(void* h);              /* number of coordinates n                        */
int32_t qaut_leon_num_codewords(void* h);  /* codewords in the spanning weight classes used  */
int64_t qaut_leon_num_incidences(void* h); /* support incidences (graph edges)                */
int32_t qaut_leon_num_classes(void* h);    /* number of ascending weight classes (colours)   */
int32_t qaut_leon_num_gens(void* h);       /* number of generators of Aut(C)                 */
int32_t qaut_leon_num_factors(void* h);    /* number of order factors (|Aut| = product)      */
int32_t qaut_leon_selector(void* h);       /* actual selector: 0=minweight, 1=congruence,
                                             3=support-minimal                             */
int32_t qaut_leon_modulus(void* h);        /* q for wt == residue (mod q), or 0 for prefix   */
int32_t qaut_leon_residue(void* h);        /* selected residue, or 0 for prefix              */
int64_t qaut_leon_enumeration_ns(void* h); /* native codeword-selection time                 */
int64_t qaut_leon_search_ns(void* h);      /* native incidence backtracking time             */
void    qaut_leon_copy_gens(void* h, int32_t* buf);    /* num_gens x n image lists (row-major)*/
void    qaut_leon_copy_factors(void* h, int64_t* buf); /* num_factors order factors           */
void    qaut_leon_copy_weights(void* h, int32_t* buf); /* num_classes ascending weights       */
void    qaut_leon_free(void* h);

/* ------------------------------------------- Brouwer-Zimmermann low-weight enumerator */
/* Collect every DISTINCT codeword of weight <= keep_weight obtained by XORing each
 * size-(1..p) subset of the rows of the (m x n) systematic generator matrix G, deduped.
 * This is the inner kernel of the BZ complete-low-weight-class certification.
 *   backend: 0 = cpu, 1 = gpu (CUDA/Metal if compiled & available), 2 = auto.
 *   threads: CPU worker threads (<=0 => hardware concurrency); ignored by the GPU backends.
 * Returns an opaque handle; free with codeaut_bz_free. */
void*       codeaut_bz_collect(const uint8_t* G, int32_t m, int32_t n,
                               int32_t p, int32_t keep_weight,
                               int32_t backend, int32_t threads);
int32_t     codeaut_bz_ok(void* h);        /* 1 if the run completed (handle valid)          */
int64_t     codeaut_bz_count(void* h);     /* number of distinct codewords collected         */
int64_t     codeaut_bz_combos(void* h);    /* number of subset-combinations evaluated        */
void        codeaut_bz_copy_rows(void* h, uint8_t* buf); /* count x n collected supports      */
const char* codeaut_bz_backend(void* h);   /* "cpu" / "cuda" / "metal" actually used         */
void        codeaut_bz_free(void* h);

/* "cpu", "gpu", "cuda", "metal", "auto" -> 1 if usable in this build, else 0. */
int32_t     codeaut_backend_available(const char* name);
const char* codeaut_version(void);

#ifdef __cplusplus
}
#endif

#endif /* CODEAUT_CAPI_H */
