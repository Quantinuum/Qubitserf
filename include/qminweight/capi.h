// Stable C ABI for the qminweight library (consumed from Python via ctypes).
#ifndef QMINWEIGHT_CAPI_H
#define QMINWEIGHT_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Result returned by the distance entry points.
typedef struct {
    int    distance;     // best upper bound found (-1 if undefined / no logicals)
    int    lower_bound;  // proven lower bound (== distance when proven)
    int    proven;       // 1 if exact (upper<=lower reached)
    int    levels;       // weight levels enumerated
    double seconds;      // wall time
    char   backend[16];  // backend actually used
} QMinWeightResult;

// method: "bz" or "mitm". which: 'Z', 'X', or 'M' for min(Z,X).
// backend: "auto","cpu","gpu". threads: 0 => hardware concurrency.
// max_weight: 0 => no cap. Returns 0 on success, nonzero on error.
int qminweight_css_distance(
    const uint8_t* Hx, int hx_rows, int hx_cols,
    const uint8_t* Hz, int hz_rows, int hz_cols,
    const char* method, char which, const char* backend,
    int threads, int max_weight, int verbose,
    QMinWeightResult* out);

// Minimum distance of a classical linear code with parity-check matrix H.
int qminweight_classical_distance(
    const uint8_t* H, int rows, int cols,
    const char* method, const char* backend,
    int threads, int max_weight, int verbose,
    QMinWeightResult* out);

int         qminweight_backend_available(const char* name); // 1 if usable
const char* qminweight_version(void);

#ifdef __cplusplus
}
#endif

#endif // QMINWEIGHT_CAPI_H
