#include <torch/extension.h>
#include <torch/library.h>
#include <c10/core/DispatchKey.h>

#include "group_norm.cuh"
#include "gelu.cuh"
#include "bind.h"

namespace fast_kernel {
    namespace operators {
        using namespace torch;
        void initFastKernelBindings(torch::Library &m) {
            m.def("group_norm_fp16", dispatch(c10::DispatchKey::CompositeExplicitAutograd, fast_kernel::operators::GroupNormFP16));
            m.def("fast_gelu_mul_fp16", dispatch(c10::DispatchKey::CompositeExplicitAutograd, fast_kernel::operators::FastGeluMulFP16));
        }
    } // namespace operators
} // namespace fast_kernel
