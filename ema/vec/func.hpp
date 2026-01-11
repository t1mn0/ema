#pragma once

#include <algorithm> // for: clamp
#include <cmath>     // for: sqrt, abs

#include "vec.hpp"

namespace ema {

template <types::Scalar T, size_t N>
constexpr T length(const Vec<T, N>& vec) {
    std::conditional<std::is_same_v<T, float>, float, double> sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum += vec[i] * vec[i];
    }
    return static_cast<T>(std::sqrt(sum));
}

template <types::Scalar T, size_t N>
constexpr T length_squared(const Vec<T, N>& vec) {
    T sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum += vec[i] * vec[i];
    }
    return sum;
}

template <types::Scalar T, size_t N>
bool is_identity(const Vec<T, N>& vec, T epsilon = std::numeric_limits<T>::epsilon()) {
    if constexpr (std::floating_point<T>) {
        return length_squared(vec) <= epsilon * epsilon;
    } else {
        return length_squared(vec) == T(0);
    }
}

template <types::Scalar T, size_t N>
constexpr Vec<T, N> normalize(const Vec<T, N>& vec) {
    auto len = length(vec);
    assert(len != 0, "Length of the vector for normalization must be positive");

    Vec<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = vec[i] * (T(1) / len);
    }
    return result;
}

template <types::Scalar T, size_t N>
constexpr Vec<T, N> dot(const Vec<T, N>& a, const Vec<T, N>& b) {
    T result = T(0);
    for (size_t i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

template <types::Scalar T>
constexpr Vec<T, 3> cross(const Vec<T, 3>& a, const Vec<T, 3>& b) {
    return Vec<T, 3>{
        a.y() * b.z() - a.z() * b.y(),
        a.z() * b.x() - a.x() * b.z(),
        a.x() * b.y() - a.y() * b.x()};
}

template <types::Scalar T, size_t N>
constexpr auto distance(const Vec<T, N>& a, const Vec<T, N>& b) {
    return length(a - b);
}

template <types::Scalar T, size_t N>
constexpr auto distance_squared(const Vec<T, N>& a, const Vec<T, N>& b) {
    return length_squared(a - b);
}

template <types::Scalar T>
constexpr auto triple(const Vec<T, 3>& a, const Vec<T, 3>& b, const Vec<T, 3>& c) {
    return dot(a, cross(b, c)); // [a, b, c] = a · (b × c)
}

template <types::Scalar T, size_t N>
constexpr auto reflect(const Vec<T, N>& incident, const Vec<T, N>& normal) {
    return incident - normal * (T(2) * dot(incident, normal));
}

template <types::Scalar T, size_t N>
constexpr auto project(const Vec<T, N>& a, const Vec<T, N>& b) {
    // project(a to b): (a · b / |b|²) * b
    if (is_identity(b)) {
        return Vec<T, N>{};
    }

    auto b_squared_length = length_squared(b);
    return b * (dot(a, b) / b_squared_length);
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
constexpr auto clamp_length(const Vec<T, N>& vec, T max_len) {
    auto len = length(vec);
    if (len <= max_len) {
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
constexpr auto angle(const Vec<T, N>& a, const Vec<T, N>& b) {
    auto dot_product = dot(a, b);
    auto len_a = length(a);
    auto len_b = length(b);

    if (len_a == T(0) || len_b == T(0))
        return T(0);

    auto cos_angle = dot_product / (len_a * len_b);
    cos_angle = std::clamp(cos_angle, T(-1), T(1));

    return std::acos(cos_angle);
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
