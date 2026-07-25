// Verbose progress output, matching the original distfind's CLI progress format: a running
// lower bound on the distance ("Distance bound: >N") with a per-level elapsed time, then a
// final exact line ("Distance: =N") with the total elapsed time. All on stderr so
// stdout stays just the number.
#pragma once
#include <cstdio>

namespace distfind {

// Elapsed duration in the original distfind's format: integer milliseconds below one second
// ("Elapsed:[15ms]"), seconds with three decimals at or above ("Elapsed:[1.087s]").
inline void fprint_elapsed(std::FILE* f, double secs) {
    if (secs < 1.0)
        std::fprintf(f, "Elapsed:[%dms]\n", (int)(secs * 1000.0 + 0.5));
    else
        std::fprintf(f, "Elapsed:[%.3fs]\n", secs);
}

// A proven lower bound (the distance is strictly greater than `lb`) plus the elapsed
// time spent ruling out that level. `tag` distinguishes interleaved searches (e.g. "Z"
// vs "X" for a CSS code's two distances); pass "" for an undecorated line.
inline void verbose_bound(int lb, double level_secs, const char* tag = "") {
    if (*tag) std::fprintf(stderr, "%s-distance bound: >%d\n", tag, lb);
    else      std::fprintf(stderr, "Distance bound: >%d\n", lb);
    fprint_elapsed(stderr, level_secs);
}

// The exact distance, plus the total elapsed time. `tag` as in verbose_bound.
inline void verbose_final(int d, double total_secs, const char* tag = "") {
    if (*tag) std::fprintf(stderr, "%s-distance: =%d\n", tag, d);
    else      std::fprintf(stderr, "Distance: =%d\n", d);
    fprint_elapsed(stderr, total_secs);
}

} // namespace distfind
