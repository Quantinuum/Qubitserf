// Misc C ABI: version string and backend-availability query.
#include "capi.h"
#include "backend.hpp"

#include <cstring>

using namespace codeaut;

extern "C" {

const char* codeaut_version(void) { return "0.1.0"; }

int32_t codeaut_backend_available(const char* name) {
    if (!name) return 0;
    auto avail = [](Backend* b) { return b && b->available() ? 1 : 0; };
    if (std::strcmp(name, "cpu") == 0)  return 1;
    if (std::strcmp(name, "cuda") == 0) return avail(cuda_backend());
    if (std::strcmp(name, "metal") == 0) return avail(metal_backend());
    if (std::strcmp(name, "gpu") == 0)  return (avail(cuda_backend()) || avail(metal_backend())) ? 1 : 0;
    if (std::strcmp(name, "auto") == 0) return 1;
    return 0;
}

}  // extern "C"
