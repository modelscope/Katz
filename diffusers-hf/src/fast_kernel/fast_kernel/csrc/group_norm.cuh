#pragma once
#include <torch/extension.h>
#include <string>

namespace fast_kernel {
    namespace operators {
        void GroupNormFP16(torch::Tensor x,
                           torch::Tensor y,
                           double eps,
                           int64_t num_groups,
                           const std::string &activation,
                           torch::Tensor weight,
                           torch::Tensor bias,
                           bool affine,
                           bool channels_last);

    } // namespace operators
} // namespace fast_kernel
