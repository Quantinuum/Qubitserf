// Stable C ABI for the distfind library (consumed from Python via ctypes).
#ifndef DISTFIND_CAPI_H
#define DISTFIND_CAPI_H

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
} DistfindResult;

// method: "bz" or "mitm". which: 'Z', 'X', or 'M' for min(Z,X).
// backend: "auto","cpu","gpu". threads: 0 => hardware concurrency.
// Returns 0 on success, nonzero on error.
int distfind_css_distance(
    const uint8_t* Hx, int hx_rows, int hx_cols,
    const uint8_t* Hz, int hz_rows, int hz_cols,
    const char* method, char which, const char* backend,
    int threads, int verbose,
    DistfindResult* out);

// Minimum distance of a classical linear code with parity-check matrix H.
int distfind_classical_distance(
    const uint8_t* H, int rows, int cols,
    const char* method, const char* backend,
    int threads, int verbose,
    DistfindResult* out);

// Result returned by the operator-weight entry point.
typedef struct {
    int    z_weight;     // min weight of the Z-part modulo rowspace(Gz)  (-1 if undefined)
    int    x_weight;     // min weight of the X-part modulo rowspace(Gx)
    int    proven;       // 1 (MITM coset leader is exact)
    double seconds;
    char   backend[16];
} DistfindOpResult;

// Operator weight (min weight modulo the group <Gx (X-type), Gz (Z-type)>).
// z_op/x_op are length-n 0/1 vectors (Z-support and X-support; a Y sets both bits).
// Pass stabilizer generators for a stabilizer code, gauge generators for a subsystem code.
// method: "bz" or "mitm" (operator weight reduces to a DistProblem); "cc" falls back
// to "bz". Returns 0 on success, nonzero on error.
int distfind_operator_weight(
    const uint8_t* Gx, int gx_rows, int gx_cols,
    const uint8_t* Gz, int gz_rows, int gz_cols,
    const uint8_t* z_op, const uint8_t* x_op, int n,
    const char* method, const char* backend,
    int threads, int verbose,
    DistfindOpResult* out);

// Subsystem CSS dressed distance from gauge generators Gx (X-type), Gz (Z-type).
// Computes the stabilizer center internally. which: 'Z','X','M'. method: "bz"|"cc"|"mitm".
// Reuses DistfindResult. Returns 0 on success.
int distfind_subsystem_distance(
    const uint8_t* Gx, int gx_rows, int gx_cols,
    const uint8_t* Gz, int gz_rows, int gz_cols,
    const char* method, char which, const char* backend,
    int threads, int verbose,
    DistfindResult* out);

// ---- general (non-CSS) stabilizer codes, symplectic [z|x] representation --------------
//
// S is a stabilizer matrix of shape rows x (2*n) in [z | x] column order: row r is the
// Pauli with Z-support S[r,0..n) and X-support S[r,n..2n). cols must be even (== 2n).
// The distance is the minimum symplectic weight of an operator in C(S) \ rowspace(S).
//
// CSS fast path: if every row is pure-X or pure-Z the call is routed to the dedicated
// CSS Hx/Hz solvers (honouring `which`: 'Z','X','M'). For a genuinely non-CSS code the
// symplectic meet-in-the-middle search is used and `which` is ignored (a single number).
// method "bz" / "cc" fall back to "mitm" for non-CSS input (with a one-line stderr note),
// since neither has a sound symplectic generalization. Returns 0 on success.
int distfind_stabilizer_distance(
    const uint8_t* S, int rows, int cols,
    const char* method, char which, const char* backend,
    int threads, int verbose,
    DistfindResult* out);

// Dressed distance of a general (non-CSS) subsystem code from its gauge generators G
// (rows x 2n, [z|x] order). The stabilizer center is computed internally; the dressed
// distance is the min symplectic weight in C(center) \ rowspace(G). CSS fast path and the
// bz/cc->mitm fallback behave as in distfind_stabilizer_distance. Returns 0 on success.
int distfind_subsystem_stabilizer_distance(
    const uint8_t* G, int rows, int cols,
    const char* method, char which, const char* backend,
    int threads, int verbose,
    DistfindResult* out);

// Operator weight of a general Pauli modulo the group <G> (rows x 2n, [z|x] order):
// the minimum symplectic weight over the coset op + rowspace(G). `op` is a length-2n
// [z|x] vector. 0 iff op is itself in rowspace(G). Pass stabilizer generators for a
// stabilizer code, gauge generators for a subsystem code. The result is returned in
// `out->distance` (lower_bound/proven set consistently). method "bz"/"cc" -> "mitm" for
// non-CSS. Returns 0 on success.
int distfind_stabilizer_operator_weight(
    const uint8_t* G, int rows, int cols,
    const uint8_t* op, int op_len,
    const char* method, const char* backend,
    int threads, int verbose,
    DistfindResult* out);

int         distfind_backend_available(const char* name); // 1 if usable
const char* distfind_version(void);

#ifdef __cplusplus
}
#endif

#endif // DISTFIND_CAPI_H
