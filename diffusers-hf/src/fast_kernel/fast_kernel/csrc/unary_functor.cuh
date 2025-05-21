#pragma once
#include "common/scalar.cuh"

namespace fast_kernel {
    namespace operators {

        enum class UnaryOp {
            kIdentity,
            kSilu,
            kRelu,
            kLeakyRelu,
            kSigmoid,
            kSelu
        };

        template <UnaryOp unary_op, typename Dst, typename Src>
        struct UnaryFunctor;

        // kIdentity
        template <typename Dst, typename Src>
        struct UnaryFunctor<UnaryOp::kIdentity, Dst, Src> {
            DEVICE_FUNC UnaryFunctor(fast_kernel::common::Scalar attr0, fast_kernel::common::Scalar attr1) {}

            DEVICE_FUNC Dst operator()(Src src) const {
                return static_cast<Dst>(src);
            }
        };

        // kSilu
        template <typename Dst, typename Src>
        struct UnaryFunctor<UnaryOp::kSilu, Dst, Src> {
            DEVICE_FUNC UnaryFunctor(fast_kernel::common::Scalar attr0, fast_kernel::common::Scalar attr1) {}

            DEVICE_FUNC Dst operator()(Src src) const {
                return static_cast<Dst>(src / (static_cast<Src>(1) + exp(-src)));
            }
        };

        // kRelu
        template <typename Dst, typename Src>
        struct UnaryFunctor<UnaryOp::kRelu, Dst, Src> {
            DEVICE_FUNC UnaryFunctor(fast_kernel::common::Scalar attr0, fast_kernel::common::Scalar attr1) {}

            DEVICE_FUNC Dst operator()(Src src) const {
                const Src zero_val = static_cast<Src>(0.0);
                if (src <= zero_val) {
                    return static_cast<Dst>(zero_val);
                }
                else {
                  return static_cast<Dst>(src);
                }
            }
        };

        // kLeakyRelu
        template <typename Dst, typename Src>
        struct UnaryFunctor<UnaryOp::kLeakyRelu, Dst, Src> {
            DEVICE_FUNC UnaryFunctor(fast_kernel::common::Scalar attr0, fast_kernel::common::Scalar attr1) : alpha(attr0.Value<float>()) {}

            DEVICE_FUNC Dst operator()(Src src) const {
                return static_cast<Dst>((src > static_cast<Src>(0.0)) ? src : alpha * src);
            }
            const Src alpha;
        };

        // kSigmoid
        template <typename Dst, typename Src>
        struct UnaryFunctor<UnaryOp::kSigmoid, Dst, Src> {
            DEVICE_FUNC UnaryFunctor(fast_kernel::common::Scalar attr0, fast_kernel::common::Scalar attr1) {}

            DEVICE_FUNC Dst operator()(Src src) const {
                return static_cast<Dst>(static_cast<Src>(1.0) / (static_cast<Src>(1.0) + exp(-src)));
            }
        };

        // kSelu
        template <typename Dst, typename Src>
        struct UnaryFunctor<UnaryOp::kSelu, Dst, Src> {
            DEVICE_FUNC UnaryFunctor(fast_kernel::common::Scalar attr0, fast_kernel::common::Scalar attr1) {}

            DEVICE_FUNC Dst operator()(Src src) const {
                return static_cast<Dst>((src > static_cast<Src>(0.0)) ? src * scale : scale * alpha * (exp(src) - static_cast<Src>(1)));
            }
            const Src scale = 1.0507009873554804934193349852946;
            const Src alpha = 1.6732632423543772848170429916717;
        };

    } // namespace operators
} // namespace fast_kernel
