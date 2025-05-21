#pragma once
#include "common/elementwise.cuh"
#include "c10/cuda/CUDAStream.h"
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <fstream>
#include <iostream>

namespace fast_kernel {
    namespace operators {
        template <typename T>
        struct FusedFastGeluMulFunctor
        {
            static constexpr T alpha = static_cast<T>(0.7978845608028654);
            static constexpr T beta = static_cast<T>(0.044714998453855515);

            FusedFastGeluMulFunctor() {}

            __device__ __forceinline__ T operator()(T x, T m) const {
                // ref to UnaryFunctor of kFastGelu
                const T half = static_cast<T>(0.5);
                const T one = static_cast<T>(1);
                const T tanh_in = alpha * (x + beta * x * x * x);
                return half * x * (one + tanh(tanh_in)) * m;
            }
        };

        template <>
        struct FusedFastGeluMulFunctor<torch::Half> {
            static constexpr float alpha = static_cast<float>(0.7978845608028654);
            static constexpr float beta = static_cast<float>(0.044714998453855515);

            FusedFastGeluMulFunctor() {}

            __device__ __forceinline__ torch::Half operator()(const torch::Half x, const torch::Half m) const {
                const torch::Half half = static_cast<torch::Half>(0.5);
                const torch::Half one = static_cast<torch::Half>(1);
                const float tanh_in = static_cast<torch::Half>(alpha) * (x + static_cast<torch::Half>(beta) * x * x * x);
                return half * x * (one + static_cast<torch::Half>(tanh(tanh_in))) * m;
            }
        };

        torch::Tensor FastGeluMulFP16(torch::Tensor x, torch::Tensor m) {
            TORCH_CHECK(x.sizes() == m.sizes(), "Input tensors x and m must have the same shape.");
            TORCH_CHECK(x.is_contiguous(), "Input tensor x must be contiguous.");
            TORCH_CHECK(m.is_contiguous(), "Input tensor m must be contiguous.");
            torch::Tensor output = torch::empty_like(x);
            const int64_t elem_cnt = x.numel();
            auto cuda_error = fast_kernel::common::Binary(
                fast_kernel::operators::FusedFastGeluMulFunctor<torch::Half>(),
                elem_cnt,
                output.data_ptr<torch::Half>(),
                x.data_ptr<torch::Half>(),
                m.data_ptr<torch::Half>(),
                at::cuda::getCurrentCUDAStream().stream());
            TORCH_CHECK(cuda_error == cudaSuccess, "CUDA error: ", cudaGetErrorString(cuda_error));
            return output;
        }

        // torch::Tensor FastGeluMulFP16(torch::Tensor x, torch::Tensor m) {
        //     TORCH_CHECK(x.sizes() == m.sizes(), "Input tensors x and m must have the same size");
        //     torch::Tensor output = torch::empty_like(x);
        //     return output;
        // }
    } // namespace operators
} // namespace fast_kernel
