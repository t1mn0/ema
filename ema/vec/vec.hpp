#pragma once

#include <cstddef> // for: size_t

#include "ema/concepts/types/scalar.hpp"

namespace ema {

template <types::Scalar T, size_t N> requires(N > 0 && N <= 4)
class alignas(sizeof(T) * (N == 3 ? 4 : N)) Vec {
  private: // member-fields:
    T components[N == 3 ? 4 : N];

  public: // member-functions:
    // default constructor:
    constexpr Vec() : components{T(0)} {}

    constexpr Vec(T num) {
        for (int i = 0; i < N; ++i) {
            components[i] = num;
        }
    }

    // Addition Identity Matrix:
    constexpr static Vec<T, N> Identity() {
        Vec<T, N> identity;
        return identity;
    }

    constexpr static Vec<T, N> Unit(size_t axis = 0) {
        Vec<T, N> unit;
        unit[axis % N] = 1;
        return unit;
    }

    // whole lotta syntax sugar but it is math:
    constexpr static Vec<T, N> Zero() {
        Vec<T, N> zero;
        return zero;
    }

    // basic general constructor:
    template <typename... Args>
    constexpr Vec(Args... args) : components{static_cast<T>(args)...} {
        static_assert(sizeof...(Args) == N, "Vec: invalid number of arguments");
    }

    constexpr size_t dimension() const { return N; }

    // clang-format off
    constexpr       T* begin()       noexcept { return components; }
    constexpr const T* begin() const noexcept { return components; }
    constexpr       T* end()         noexcept { return components + N; }
    constexpr const T* end()   const noexcept { return components + N; }
    // clang-format on

    Vec operator-() const {
        Vec<T, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.components[i] = -components[i];
        }
        return result;
    }

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
template <types::Scalar T, size_t N>
constexpr Vec<T, N> operator+(Vec<T, N> lhs, const Vec<T, N>& rhs) {
    lhs += rhs;
    return lhs;
}

// free binary operators:
template <types::Scalar T, size_t N>
constexpr Vec<T, N> operator-(Vec<T, N> lhs, const Vec<T, N>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <types::Scalar T, size_t N>
constexpr Vec<T, N> operator*(Vec<T, N> vec, T scalar) {
    vec *= scalar;
    return vec;
}

template <types::Scalar T, size_t N>
constexpr Vec<T, N> operator*(T scalar, Vec<T, N> vec) {
    return vec * scalar;
}

template <types::Scalar T, size_t N>
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

#include "func.hpp"
