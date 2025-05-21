#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cutlass/fast_math.h>
#include <cuda_fp16.h>
#include "c10/cuda/CUDAStream.h"

#include "unary_functor.cuh"
#include "normalize.cuh"
#include "group_norm.cuh"

using namespace fast_kernel::operators;

namespace fast_kernel {
    namespace operators {

        template <typename SRC, typename DST, UnaryOp activation, bool affine>
        struct AffineStore {
            AffineStore(DST *y, int64_t row_size, int64_t channel_size, int64_t spatial_size,
                        const DST *gamma, const DST *beta)
                      : y(y),
                        row_size(row_size),
                        channel_size(channel_size),
                        spatial_size(spatial_size),
                        gamma(gamma),
                        beta(beta),
                        act(0, 0) {}

            template <int PackSize>
            __device__ void store(const SRC *src, int64_t row, int64_t col) {
                Pack<DST, PackSize> y_pack;
                const int64_t offset = row * row_size + col;
                const int64_t packed_offset = offset / PackSize;
                const int64_t gamma_beta_offset = (offset / spatial_size) % channel_size;
                DST gamma_val = 1.0;
                DST beta_val = 0.0;
                if (affine)
                {
                  gamma_val = gamma[gamma_beta_offset];
                  beta_val = beta[gamma_beta_offset];
                }

                #pragma unroll
                for (int i = 0; i < PackSize; ++i)
                {
                    DST normalized_i = static_cast<DST>(src[i]);
                    if (affine) {
                      y_pack.elem[i] = act(normalized_i * gamma_val + beta_val);
                    }
                    else {
                      // Direct Store.
                      y_pack.elem[i] = act(normalized_i);
                    }
                }
                *(reinterpret_cast<PackType<DST, PackSize> *>(y) + packed_offset) =
                    y_pack.storage;
            }
            bool CanPackAs(size_t pack_size) { return (spatial_size % pack_size) == 0; }
            DST *y;
            int64_t row_size;
            int64_t channel_size;
            int64_t spatial_size;
            const DST *gamma;
            const DST *beta;
            UnaryFunctor<activation, DST, DST> act;
        };

        template <typename SRC, typename DST, UnaryOp activation, bool affine>
        struct ChannelsLastStore {
            ChannelsLastStore(DST *y, const DST *gamma, const DST *beta, int64_t spatial_size,
                              int64_t channel_size, int64_t num_groups)
                            : y(y),
                              gamma(gamma),
                              beta(beta),
                              spatial_size(spatial_size),
                              c0(num_groups),
                              c1(channel_size / num_groups),
                              act(0, 0) {}

            template <int PackSize>
            __device__ void store(const SRC *src, int32_t row, int32_t col) {
                Pack<DST, PackSize> y_pack;
                Pack<DST, PackSize> gamma_pack;
                Pack<DST, PackSize> beta_pack;
                int32_t spatial_idx;
                int32_t c1_idx;
                c1(spatial_idx, c1_idx, col);
                int32_t batch_idx;
                int32_t c0_idx;
                c0(batch_idx, c0_idx, row);
                const int32_t y_offset =
                    (batch_idx * c0.divisor * c1.divisor * spatial_size + spatial_idx * c0.divisor * c1.divisor + c0_idx * c1.divisor + c1_idx) / PackSize;
                const int32_t gamma_beta_offset = (c0_idx * c1.divisor + c1_idx) / PackSize;
                if (affine) {
                    gamma_pack.storage =
                        *(reinterpret_cast<const PackType<DST, PackSize> *>(gamma) + gamma_beta_offset);
                    beta_pack.storage = *(reinterpret_cast<const PackType<DST, PackSize> *>(beta) + gamma_beta_offset);
                }

                #pragma unroll
                for (int i = 0; i < PackSize; ++i) {
                    DST normalized_i = static_cast<DST>(src[i]);
                    if (affine) {
                      y_pack.elem[i] = act(normalized_i * gamma_pack.elem[i] + beta_pack.elem[i]);
                    }
                    else {
                      // Direct Store.
                      y_pack.elem[i] = act(normalized_i);
                    }
                }
                *(reinterpret_cast<PackType<DST, PackSize> *>(y) + y_offset) = y_pack.storage;
            }
            bool CanPackAs(size_t pack_size) { return (c1.divisor % pack_size) == 0; }
            DST *y;
            const DST *gamma;
            const DST *beta;
            int32_t spatial_size;
            cutlass::FastDivmod c0;
            cutlass::FastDivmod c1;
            UnaryFunctor<activation, DST, DST> act;
        };

        template <typename SRC, typename DST>
        struct ChannelsLastLoad {
            using LoadType = DST;
            ChannelsLastLoad(const SRC *src, int64_t spatial_size, int64_t channel_size, int64_t num_groups)
                : src(src), spatial_size(spatial_size), c0(num_groups), c1(channel_size / num_groups) {}
            template <int N>
            __device__ void load(DST *dst, int32_t row, int32_t col) const
            {
              int32_t spatial_idx;
              int32_t c1_idx;
              c1(spatial_idx, c1_idx, col);
              int32_t batch_idx;
              int32_t c0_idx;
              c0(batch_idx, c0_idx, row);
              Pack<SRC, N> pack;
              const int32_t offset = (batch_idx * c0.divisor * c1.divisor * spatial_size + spatial_idx * c0.divisor * c1.divisor + c0_idx * c1.divisor + c1_idx) / N;       
              pack.storage = *(reinterpret_cast<const PackType<SRC, N> *>(src) + offset);
              #pragma unroll
              for (int i = 0; i < N; ++i) {
                dst[i] = static_cast<DST>(pack.elem[i]);
              }
            }
            bool CanPackAs(size_t pack_size) { return (c1.divisor % pack_size) == 0; }
            const SRC *src;
            int32_t spatial_size;
            cutlass::FastDivmod c0;
            cutlass::FastDivmod c1;
        };

        template <typename T, UnaryOp activation, bool affine>
        void GroupNormForwardGPU(at::cuda::CUDAStream *stream, const int64_t num_instances, const int64_t norm_size,
                                 const int64_t channel_size, const int64_t spatial_size,
                                 const double epsilon, const T *x_ptr, const T *gamma_ptr,
                                 const T *beta_ptr, T *y_ptr, torch::Tensor mean,
                                 torch::Tensor inv_variance, bool channels_first) {
            using ComputeType = typename DefaultComputeType<T>::type;
            if (channels_first) {
                DirectLoad<T, T> load(x_ptr, norm_size);
                AffineStore<ComputeType, T, activation, affine> store(y_ptr, norm_size, channel_size,
                                                                      spatial_size, gamma_ptr, beta_ptr);

                DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
                    stream->stream(), load, store, num_instances, norm_size, epsilon,
                    mean.data_ptr<ComputeType>(), inv_variance.data_ptr<ComputeType>());
            }
            else {
                ChannelsLastLoad<T, T> load(x_ptr, spatial_size, channel_size,
                                            channel_size / (norm_size / spatial_size));
                ChannelsLastStore<ComputeType, T, activation, affine> store(
                    y_ptr, gamma_ptr, beta_ptr, spatial_size, channel_size,
                    channel_size / (norm_size / spatial_size));

                DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
                    stream->stream(), load, store, num_instances, norm_size, epsilon,
                    mean.data_ptr<ComputeType>(), inv_variance.data_ptr<ComputeType>());
            }
        }

        template <typename T, UnaryOp activation>
        void DispatchGroupNormAffine(at::cuda::CUDAStream *stream, const int64_t num_instances,
                                     const int64_t norm_size, const int64_t channel_size,
                                     const int64_t spatial_size, const double epsilon, const T *x_ptr,
                                     const T *gamma_ptr, const T *beta_ptr, T *y_ptr, torch::Tensor mean,
                                     torch::Tensor inv_variance, bool channels_first) {
            if (gamma_ptr != nullptr && beta_ptr != nullptr) {
                GroupNormForwardGPU<T, activation, true>(stream, num_instances, norm_size, channel_size,
                                                         spatial_size, epsilon, x_ptr, gamma_ptr, beta_ptr,
                                                         y_ptr, mean, inv_variance, channels_first);
            }
            else {
                GroupNormForwardGPU<T, activation, false>(stream, num_instances, norm_size, channel_size,
                                                          spatial_size, epsilon, x_ptr, gamma_ptr, beta_ptr,
                                                          y_ptr, mean, inv_variance, channels_first);
            }
        }

        template <typename T>
        void DispatchGroupNormForwardGPU(at::cuda::CUDAStream *stream, const int64_t num_instances,
                                         const int64_t norm_size, const int64_t channel_size,
                                         const int64_t spatial_size, const double epsilon, const T *x_ptr,
                                         const T *gamma_ptr, const T *beta_ptr, T *y_ptr,
                                         torch::Tensor mean, torch::Tensor inv_variance,
                                         bool channels_first, const std::string &activation) {
            if (activation == "none") {
                DispatchGroupNormAffine<T, UnaryOp::kIdentity>(
                    stream, num_instances, norm_size, channel_size, spatial_size, epsilon, x_ptr, gamma_ptr,
                    beta_ptr, y_ptr, mean, inv_variance, channels_first);
            }
            else if (activation == "silu") {
                DispatchGroupNormAffine<T, UnaryOp::kSilu>(
                    stream, num_instances, norm_size, channel_size, spatial_size, epsilon, x_ptr, gamma_ptr,
                    beta_ptr, y_ptr, mean, inv_variance, channels_first);
            }
            else {
              printf("Unsupported activation type: %s\n", activation.c_str());
            }
        }

        template <typename T>
        void GroupNorm(torch::Tensor x,
                       torch::Tensor y,
                       double eps,
                       int64_t num_groups,
                       const std::string &activation,
                       torch::Tensor weight,
                       torch::Tensor bias,
                       bool affine,
                       bool channels_first) {

            const auto x_shape = x.sizes();
            const int batch_size = x_shape[0];
            int64_t channel_size = x_shape[1];

            // mean and variance
            auto options = torch::TensorOptions().dtype(torch::kFloat).device(x.device());
            auto mean = torch::empty({batch_size, num_groups}, options);
            auto inv_variance = torch::empty({batch_size, num_groups}, options);

            const int64_t num_instances = mean.numel();                           // batch_size * num_groups
            const int64_t norm_size = x.numel() / num_instances;                  // CHW / num_groups
            const int64_t spatial_size = x.numel() / (batch_size * channel_size); // H*W

            // call
            auto stream = at::cuda::getCurrentCUDAStream();
            if (affine == true) {
                DispatchGroupNormForwardGPU<T>(
                    &stream, num_instances, norm_size, channel_size,
                    spatial_size, eps, x.data_ptr<T>(), weight.data_ptr<T>(), bias.data_ptr<T>(),
                    y.data_ptr<T>(), mean, inv_variance, channels_first, activation);
            }
            else {
                DispatchGroupNormForwardGPU<T>(
                    &stream, num_instances, norm_size, channel_size,
                    spatial_size, eps, x.data_ptr<T>(), nullptr, nullptr,
                    y.data_ptr<T>(), mean, inv_variance, channels_first, activation);
            }
        }

        void GroupNormFP16(torch::Tensor x,
                           torch::Tensor y,
                           double eps,
                           int64_t num_groups,
                           const std::string &activation,
                           torch::Tensor weight,
                           torch::Tensor bias,
                           bool affine,
                           bool channels_last) {
            GroupNorm<torch::Half>(x, y, eps, num_groups, activation, weight, bias, affine, channels_last);
        }

    } // namespace operators
} // namespace fast_kernel
