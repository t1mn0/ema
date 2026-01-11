#pragma once

#include <cmath>

#include "mat.hpp"

namespace ema {

// Basic arithmetic operators:
template <types::Scalar T, size_t C, size_t R>
constexpr Mat<T, C, R> operator-(const Mat<T, C, R>& mat) {
    Mat<T, C, R> result;
    for (size_t i = 0; i < C; ++i) {
        result.col(i) = -mat.col(i);
    }
    return result;
}

template <types::Scalar T, size_t C, size_t R>
constexpr Mat<T, C, R> operator+(Mat<T, C, R> lhs, const Mat<T, C, R>& rhs) {
    lhs += rhs;
    return lhs;
}

template <types::Scalar T, size_t C, size_t R>
constexpr Mat<T, C, R> operator-(Mat<T, C, R> lhs, const Mat<T, C, R>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <types::Scalar T, size_t C, size_t R>
constexpr Mat<T, C, R> operator*(Mat<T, C, R> mat, T scalar) {
    mat *= scalar;
    return mat;
}

template <types::Scalar T, size_t C, size_t R>
constexpr Mat<T, C, R> operator*(T scalar, Mat<T, C, R> mat) {
    return mat * scalar;
}

template <types::Scalar T, size_t C, size_t R>
constexpr Mat<T, C, R> operator/(Mat<T, C, R> mat, T scalar) {
    for (size_t i = 0; i < C; ++i) {
        mat.col(i) /= scalar;
    }
    return mat;
}

template <types::Scalar T, size_t C, size_t R>
constexpr Vec<T, R> operator*(const Mat<T, C, R>& mat, const Vec<T, C>& vec) {
    // Mat<R, C> * Vec<C> = Mat<R, C> * Mat<C,1> = Mat<R,1> = Vec<R>
    Vec<T, R> result = Vec<T, R>::Zero();

    for (size_t c = 0; c < C; ++c) {
        result += mat.col(c) * vec[c];
    }

    return result;
}

template <types::Scalar T, size_t C, size_t R>
constexpr Vec<T, C> operator*(const Vec<T, R>& vec, const Mat<T, C, R>& mat) {
    // Vec<R> * Mat<R, C> = Mat<1, R> * Mat<R, C> = Mat<1, C> = Vec<C>
    Vec<T, C> result = Vec<T, C>::Zero();

    for (size_t r = 0; r < R; ++r) {
        result += mat.columns[r] * vec[r];
    }

    return result;
}

// Linear algebra functions:
template <types::Scalar T, size_t C, size_t R>
constexpr Mat<T, R, C> transpose(const Mat<T, C, R>& mat) {
    Mat<T, R, C> result;
    for (size_t r = 0; r < R; ++r) {
        for (size_t c = 0; c < C; ++c) {
            result(c, r) = mat(r, c);
        }
    }
    return result;
}

template <types::Scalar T, size_t N>
constexpr T trace(const Mat<T, N, N>& mat) {
    T sum = T(0);
    for (size_t i = 0; i < N; ++i) {
        sum += mat(i, i);
    }
    return sum;
}

template <types::Scalar T>
constexpr T determinant(const Mat<T, 2, 2>& mat) {
    return mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
}

template <types::Scalar T>
constexpr T determinant(const Mat<T, 3, 3>& mat) {
    return mat(0, 0) * mat(1, 1) * mat(2, 2) +
           mat(0, 1) * mat(1, 2) * mat(2, 0) +
           mat(0, 2) * mat(1, 0) * mat(2, 1) -
           mat(0, 2) * mat(1, 1) * mat(2, 0) -
           mat(0, 1) * mat(1, 0) * mat(2, 2) -
           mat(0, 0) * mat(1, 2) * mat(2, 1);
}

template <types::Scalar T, size_t N>
constexpr T minor(const Mat<T, N, N>& mat, size_t row, size_t col) {
    static_assert(N > 1, "Minor requires at least 2x2 matrix");

    Mat<T, N - 1, N - 1> submatrix;
    size_t sub_i = 0, sub_j = 0;

    for (size_t i = 0; i < N; ++i) {
        if (i == row)
            continue;
        sub_j = 0;
        for (size_t j = 0; j < N; ++j) {
            if (j == col)
                continue;
            submatrix(sub_i, sub_j) = mat(i, j);
            ++sub_j;
        }
        ++sub_i;
    }

    return determinant(submatrix);
}

template <types::Scalar T, size_t N>
constexpr T cofactor(const Mat<T, N, N>& mat, size_t row, size_t col) {
    T minor_val = minor(mat, row, col);
    return ((row + col) % 2 == 0) ? minor_val : -minor_val;
}

template <types::Scalar T, size_t N>
constexpr T determinant(const Mat<T, N, N>& mat) {
    if constexpr (N == 1) {
        return mat(0, 0);
    } else {
        T det = T(0);
        for (size_t j = 0; j < N; ++j) {
            det += mat(0, j) * cofactor(mat, 0, j);
        }
        return det;
    }
}

template <types::Scalar T, size_t N>
constexpr Mat<T, N, N> inverse(const Mat<T, N, N>& mat) {
    constexpr T tolerance = static_cast<T>(1e-10);

    T det = determinant(mat);
    if (std::abs(det) < tolerance) {
        return Mat<T, N, N>{};
    }

    Mat<T, N, N> result;
    T inv_det = T(1) / det;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(j, i) = cofactor(mat, i, j) * inv_det;
        }
    }

    return result;
}

template <types::Scalar T>
constexpr Mat<T, 4, 4> translation(const Vec<T, 3>& v) {
    auto res = Mat<T, 4, 4>::Identity();
    res.col(3) = Vec<T, 4>{v.x(), v.y(), v.z(), T(1)};
    return res;
}

template <types::Scalar T>
constexpr Mat<T, 4, 4> look_at(const Vec<T, 3>& eye, const Vec<T, 3>& target, const Vec<T, 3>& up) {
    auto z = normalize(eye - target); // Forward
    auto x = normalize(cross(up, z)); // Right
    auto y = cross(z, x);             // Up

    Mat<T, 4, 4> res = Mat<T, 4, 4>::Identity();
    res(0, 0) = x.x();
    res(0, 1) = x.y();
    res(0, 2) = x.z();
    res(1, 0) = y.x();
    res(1, 1) = y.y();
    res(1, 2) = y.z();
    res(2, 0) = z.x();
    res(2, 1) = z.y();
    res(2, 2) = z.z();

    res(0, 3) = -dot(x, eye);
    res(1, 3) = -dot(y, eye);
    res(2, 3) = -dot(z, eye);
    return res;
}

template <types::Scalar T>
constexpr Mat<T, 4, 4> perspective(T fov_rad, T aspect, T near, T far) {
    T tan_half_fov = std::tan(fov_rad / T(2));
    Mat<T, 4, 4> res = Mat<T, 4, 4>::Zero();
    res(0, 0) = T(1) / (aspect * tan_half_fov);
    res(1, 1) = T(1) / tan_half_fov;
    res(2, 2) = -(far + near) / (far - near);
    res(2, 3) = -(T(2) * far * near) / (far - near);
    res(3, 2) = T(-1);
    return res;
}

template <types::Scalar T>
constexpr Mat<T, 4, 4> ortho(T left, T right, T bottom, T top, T near, T far) {
    auto res = Mat<T, 4, 4>::Identity();
    res(0, 0) = T(2) / (right - left);
    res(1, 1) = T(2) / (top - bottom);
    res(2, 2) = -T(2) / (far - near);
    res(0, 3) = -(right + left) / (right - left);
    res(1, 3) = -(top + bottom) / (top - bottom);
    res(2, 3) = -(far + near) / (far - near);
    return res;
}

// Special matrices:
template <types::Scalar T>
constexpr Mat<T, 2, 2> rotation_matrix_2d(T angle_rad) {
    T c = std::cos(angle_rad);
    T s = std::sin(angle_rad);

    return Mat<T, 2, 2>{
        Vec<T, 2>{c, -s},
        Vec<T, 2>{s, c}};
}

template <types::Scalar T>
constexpr Mat<T, 3, 3> rotation_matrix_x(T angle_rad) {
    T c = std::cos(angle_rad);
    T s = std::sin(angle_rad);

    return Mat<T, 3, 3>{
        Vec<T, 3>{1, 0, 0},
        Vec<T, 3>{0, c, -s},
        Vec<T, 3>{0, s, c}};
}

template <types::Scalar T>
constexpr Mat<T, 3, 3> rotation_matrix_y(T angle_rad) {
    T c = std::cos(angle_rad);
    T s = std::sin(angle_rad);

    return Mat<T, 3, 3>{
        Vec<T, 3>{c, 0, s},
        Vec<T, 3>{0, 1, 0},
        Vec<T, 3>{-s, 0, c}};
}

template <types::Scalar T>
constexpr Mat<T, 3, 3> rotation_matrix_z(T angle_rad) {
    T c = std::cos(angle_rad);
    T s = std::sin(angle_rad);

    return Mat<T, 3, 3>{
        Vec<T, 3>{c, -s, 0},
        Vec<T, 3>{s, c, 0},
        Vec<T, 3>{0, 0, 1}};
}

template <types::Scalar T>
constexpr Mat<T, 3, 3> scaling_matrix_3d(T sx, T sy, T sz) {
    return Mat<T, 3, 3>{
        Vec<T, 3>{sx, 0, 0},
        Vec<T, 3>{0, sy, 0},
        Vec<T, 3>{0, 0, sz}};
}

template <types::Scalar T>
constexpr Mat<T, 3, 3> scaling_matrix_3d(T scale) {
    return scaling_matrix_3d(scale, scale, scale);
}

// Utility functions:
template <types::Scalar T, size_t N>
constexpr bool is_symmetric(const Mat<T, N, N>& mat) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            if (mat(i, j) != mat(j, i)) {
                return false;
            }
        }
    }
    return true;
}

// Equality comparator operators:
template <types::Scalar T, size_t C, size_t R>
constexpr bool operator==(const Mat<T, C, R>& a, const Mat<T, C, R>& b) {
    return are_equal(a, b, T(0));
}

template <types::Scalar T, size_t C, size_t R>
constexpr bool operator!=(const Mat<T, C, R>& a, const Mat<T, C, R>& b) {
    return !(a == b);
}

} // namespace ema
