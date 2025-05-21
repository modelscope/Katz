#include "scalar.cuh"

namespace fast_kernel {
    namespace common{

        #define DEFINE_SCALAR_BINARY_OP(op)                                               \
        Scalar& Scalar::operator op##=(const Scalar& other) {                             \
            if (IsComplex() || other.IsComplex()) {                                       \
                std::complex<double> val =                                                \
                    Value<std::complex<double>>() op other.Value<std::complex<double>>(); \
                *this = val;                                                              \
            }                                                                             \
            if (IsFloatingPoint() || other.IsFloatingPoint()) {                           \
                double val = As<double>() op other.As<double>();                          \
                *this = val;                                                              \
            } else {                                                                      \
                int64_t val = As<int64_t>() op other.As<int64_t>();                       \
                *this = val;                                                              \
            }                                                                             \
            return *this;                                                                 \
        }                                                                                 \
        Scalar Scalar::operator op(const Scalar& other) const {                           \
            if (IsComplex() || other.IsComplex()) {                                       \
                std::complex<double> val =                                                \
                    Value<std::complex<double>>() op other.Value<std::complex<double>>(); \
                return Scalar(val);                                                       \
            }                                                                             \
            if (IsFloatingPoint() || other.IsFloatingPoint()) {                           \
                double val = As<double>() op other.As<double>();                          \
                return Scalar(val);                                                       \
            }                                                                             \
            int64_t val = As<int64_t>() op other.As<int64_t>();                           \
            return Scalar(val);                                                           \
        }

        DEFINE_SCALAR_BINARY_OP(+);
        DEFINE_SCALAR_BINARY_OP(-);
        DEFINE_SCALAR_BINARY_OP(*);
        DEFINE_SCALAR_BINARY_OP(/);  // NOLINT

        #undef DEFINE_SCALAR_BINARY_OP


    } // namespace common
} // namespace fast_kernel
