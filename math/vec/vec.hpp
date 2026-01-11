#pragma once

#include <cstddef> // for: size_t

#include "../concepts/scalar.hpp"

namespace ema {

template <concepts::Scalar T, size_t N> requires(N > 0 && N <= 4)
class alignas(sizeof(T) * (N == 3 ? 4 : N)) Vec {
  public: // member-fields:
    T components[N == 3 ? 4 : N];

  public: // member-functions:
    // default constructor:
    constexpr Vec() : components{T(0)} {}

    constexpr static Vec<T, N> AdditionIdentity() {
        Vec<T, N> identity;
        return identity;
    }

    // basic general constructor:
    template <typename... Args>
    constexpr Vec(Args... args) : components{static_cast<T>(args)...} {
        static_assert(sizeof...(Args) == N, "Vec: invalid number of arguments");
    }

    // clang-format off
    constexpr       T* begin()       noexcept { return components; }
    constexpr const T* begin() const noexcept { return components; }
    constexpr       T* end()         noexcept { return components + N; }
    constexpr const T* end()   const noexcept { return components + N; }
    // clang-format on

    Vec& operator+=(const Vec& other) {
        for (size_t i = 0; i < N; ++i) {
            components[i] += other.components[i];
        }
        return *this;
    }

    Vec& operator-=(const Vec& other) {
        for (size_t i = 0; i < N; ++i) {
            components[i] -= other.components[i];
        }
        return *this;
    }

    Vec& operator*=(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            components[i] *= scalar;
        }
        return *this;
    }

    Vec& operator/=(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            components[i] /= scalar;
        }
        return *this;
    }

    // clang-format off
    constexpr       T& operator[](size_t i)       { return components[i]; }
    constexpr const T& operator[](size_t i) const { return components[i]; }

    constexpr       auto& x()       requires(N >= 1) { return components[0]; }
    constexpr const auto& x() const requires(N >= 1) { return components[0]; }
    constexpr       auto& y()       requires(N >= 2) { return components[1]; }
    constexpr const auto& y() const requires(N >= 2) { return components[1]; }
    constexpr       auto& z()       requires(N >= 3) { return components[2]; }
    constexpr const auto& z() const requires(N >= 3) { return components[2]; }
    constexpr       auto& w()       requires(N >= 4) { return components[3]; }
    constexpr const auto& w() const requires(N >= 4) { return components[3]; }
    // clang-format on
};

// free binary and unary operators:
template <concepts::Scalar T, size_t N>
constexpr Vec<T, N> operator+(Vec<T, N> lhs, const Vec<T, N>& rhs) {
    lhs += rhs;
    return lhs;
}

// free binary operators:
template <concepts::Scalar T, size_t N>
constexpr Vec<T, N> operator-(Vec<T, N> lhs, const Vec<T, N>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <concepts::Scalar T, size_t N>
constexpr Vec<T, N> operator*(Vec<T, N> vec, T scalar) {
    vec *= scalar;
    return vec;
}

template <concepts::Scalar T, size_t N>
constexpr Vec<T, N> operator*(T scalar, Vec<T, N> vec) {
    return vec * scalar;
}

template <concepts::Scalar T, size_t N>
constexpr Vec<T, N> operator/(Vec<T, N> lhs, T scalar) {
    lhs /= scalar;
    return lhs;
}

// aliases:
using Vec2i = Vec<int, 2>;
using Vec3i = Vec<int, 3>;
using Vec4i = Vec<int, 4>;
using Vec2f = Vec<float, 2>;
using Vec3f = Vec<float, 3>;
using Vec4f = Vec<float, 4>;
using Vec2d = Vec<double, 2>;
using Vec3d = Vec<double, 3>;
using Vec4d = Vec<double, 4>;

} // namespace ema
