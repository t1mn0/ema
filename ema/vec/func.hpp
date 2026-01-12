#pragma once

#include <algorithm> // for: clamp
#include <cassert>
#include <cmath> // for: sqrt, abs, acos

#include "ema/angle/angle.hpp"
#include "vec.hpp"

namespace ema {

template <types::Scalar T, size_t N>
constexpr T is_zero(const Vec<T, N>& vec) {
    return len(vec) == T(0);
}

template <types::Scalar T, size_t N>
constexpr std::conditional_t<std::is_same_v<T, double>, double, float> len(const Vec<T, N>& vec) {
    // type(sum): if (T == double) => double; else float;
    using SumType = std::conditional_t<std::is_same_v<T, double>, double, float>;
    SumType sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

template <types::Scalar T, size_t N>
constexpr T len_squared(const Vec<T, N>& vec) {
    T sum = T(0);
    for (size_t i = 0; i < N; ++i) {
        sum += vec[i] * vec[i];
    }
    return sum;
}

template <types::Scalar T, size_t N>
constexpr Vec<T, N> normalize(const Vec<T, N>& vec) {
    auto l = len(vec);
    assert(l != 0.0 && "Length of the vector for normalization must be not zero");

    Vec<T, N> norm;
    for (size_t i = 0; i < N; ++i) {
        norm[i] = vec[i] / l;
    }
    return norm;
}

template <types::Scalar T, size_t N>
constexpr T dot(const Vec<T, N>& a, const Vec<T, N>& b) {
    T sum = T(0);
    for (size_t i = 0; i < N; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

template <types::Scalar T, size_t N>
constexpr T distance(const Vec<T, N>& a, const Vec<T, N>& b) {
    return len(a - b);
}

template <types::Scalar T, size_t N>
constexpr T distance_squared(const Vec<T, N>& a, const Vec<T, N>& b) {
    return len_squared(a - b);
}

template <types::Scalar T>
constexpr Vec<T, 3> cross_product(const Vec<T, 3>& a, const Vec<T, 3>& b) {
    return Vec<T, 3>{
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x()};
}

template <types::Scalar T>
constexpr auto triple_product(const Vec<T, 3>& a, const Vec<T, 3>& b, const Vec<T, 3>& c) {
    return dot(a, cross_product(b, c)); // [a, b, c] = a · (b × c)
}

template <types::Scalar T, size_t N>
constexpr auto reflect(const Vec<T, N>& incident, const Vec<T, N>& normal) {
    return incident - normal * (T(2) * dot(incident, normal));
}

template <types::Scalar T, size_t N>
constexpr auto project(const Vec<T, N>& a, const Vec<T, N>& b) {
    // project(a to b): (a · b / |b|²) * b
    if (is_zero(b)) { // invalid argument case ([MAYBE_TODO]: handle this scenario more strictly)
        return Vec<T, N>{};
    }

    auto b_squared_len = len_squared(b);
    return b * (dot(a, b) / b_squared_len);
}

template <types::Scalar T, size_t N>
constexpr auto reject(const Vec<T, N>& a, const Vec<T, N>& b) {
    return a - project(a, b);
}

template <types::Scalar T, size_t N>
constexpr auto lerp(const Vec<T, N>& a, const Vec<T, N>& b, T t) {
    Vec<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] + t * (b[i] - a[i]);
    }
    return result;
}

template <types::Scalar T, size_t N>
constexpr auto clamp_len(const Vec<T, N>& vec, T max_len) {
    auto l = len(vec);
    if (l <= max_len) {
        return vec;
    }
    return normalize(vec) * max_len;
}

template <types::Scalar T, size_t N>
constexpr auto abs(const Vec<T, N>& vec) {
    Vec<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = std::abs(vec[i]);
    }
    return result;
}

template <types::Scalar T, size_t N>
constexpr Angle<T> angle(const Vec<T, N>& a, const Vec<T, N>& b) {
    auto dot_product = dot(a, b);
    auto len_a = len(a);
    auto len_b = len(b);

    if (len_a == T(0) || len_b == T(0))
        return Angle<T>::rad(T(0));

    auto cos_angle = dot_product / (len_a * len_b);
    cos_angle = std::clamp(cos_angle, T(-1), T(1));

    return Angle<T>::rad(static_cast<T>(std::acos(cos_angle)));
}

template <types::Scalar T, size_t N>
constexpr bool is_orthogonal(const Vec<T, N>& a, const Vec<T, N>& b, T epsilon = std::numeric_limits<T>::epsilon()) {
    if constexpr (std::floating_point<T>) {
        return std::abs(dot(a, b)) <= epsilon;
    } else {
        return dot(a, b) == T(0);
    }
}

} // namespace ema
