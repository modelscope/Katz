#include <torch/extension.h>

#include "bind.h"
using namespace fast_kernel;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // operators::initFastKernelBindings(m);
}

TORCH_LIBRARY(fast_kernel, m) {
    operators::initFastKernelBindings(m);
}
