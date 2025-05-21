#pragma once
#include <torch/extension.h>

#include <torch/library.h>

namespace fast_kernel {
    namespace operators {
        using namespace torch;
        void initFastKernelBindings(torch::Library &m);
    }
}
