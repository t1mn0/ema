#pragma once

#include "ema/vec/vec.hpp"

namespace ema {

template <types::Scalar T, size_t C, size_t R>
class Mat {
  private: // member-fields:
    // column-major order:
    Vec<T, R> columns[C];
    //    Vec      Vec     Vectors     Vec      Vec
    // <---R---><---R---><---...---><---R---><---R--->
    //     1        2        ...       C-1       C

  public: // member-functions:
    // default constructor:
    constexpr Mat() = default;

    constexpr Mat(T num) {
        for (size_t i = 0; i < C; ++i) {
            for (size_t j = 0; j < R; ++j) {
                columns[i][j] = num;
            }
        }
    }

    template <typename... VecArgs>
    requires(sizeof...(VecArgs) == C) && (std::is_same_v<VecArgs, Vec<T, R>> && ...)
    constexpr Mat(VecArgs&&... cols) : columns{cols...} {}

    // Multiplication Identity Matrix:
    constexpr static Mat Identity() requires(C == R) {
        Mat result;
        for (size_t i = 0; i < C; ++i) {
            result.columns[i][i] = T(1);
        }
        return result;
    }

    static constexpr Mat Zero() {
        return Mat{};
    }

    static constexpr Mat Diagonal(T val) requires(C == R) {
        Mat result;
        for (size_t i = 0; i < C; ++i) {
            result(i, i) = val;
        }
        return result;
    }

    constexpr bool is_square() const { return C == R; }
    constexpr size_t col_num() const { return C; }
    constexpr size_t row_num() const { return R; }

    // clang-format off
    constexpr       Vec<T, R>* begin()       noexcept { return columns; }
    constexpr const Vec<T, R>* begin() const noexcept { return columns; }
    constexpr       Vec<T, R>* end()         noexcept { return columns + C; }
    constexpr const Vec<T, R>* end()   const noexcept { return columns + C; }

    constexpr       Vec<T, R>& col(size_t index)       { return columns[index]; }
    constexpr const Vec<T, R>& col(size_t index) const { return columns[index]; }
    // clang-format on

    // atypical element access operator for matrices due to the column-oriented structure:
    constexpr T& operator()(size_t row, size_t col) {
        return columns[col][row];
    }

    constexpr const T& operator()(size_t row, size_t col) const {
        return columns[col][row];
    }

    Mat<T, C, R>& operator+=(const Mat<T, C, R>& oth) {
        for (size_t i = 0; i < C; ++i) {
            columns[i] += oth.columns[i];
        }
        return *this;
    }

    Mat& operator-=(const Mat& oth) {
        for (size_t i = 0; i < C; ++i) {
            columns[i] -= oth.columns[i];
        }
        return *this;
    }

    Mat& operator*=(T scalar) {
        for (size_t i = 0; i < C; ++i) {
            columns[i] *= scalar;
        }
        return *this;
    }

    constexpr Vec<T, R> operator*(const Vec<T, C>& vec) const {
        Vec<T, R> result = Vec<T, R>::Zero();
        for (size_t c = 0; c < C; ++c) {
            result += columns[c] * vec[c];
        }
        return result;
    }

    template <size_t Q>
    constexpr Mat<T, Q, R> operator*(const Mat<T, Q, C>& oth) const {
        // Mat<R, C> * Mat<C, Q> = Mat<R, Q>
        Mat<T, Q, R> result;
        for (size_t j = 0; j < Q; ++j) {
            Vec<T, R> res_col = columns[0] * oth(0, j);

            for (size_t i = 1; i < C; ++i) {
                res_col += columns[i] * oth(i, j);
            }

            result.col(j) = res_col;
        }
        return result;
    }
};

using Mat2i = Mat<int, 2, 2>;
using Mat3i = Mat<int, 3, 3>;
using Mat4i = Mat<int, 4, 4>;
using Mat2f = Mat<float, 2, 2>;
using Mat3f = Mat<float, 3, 3>;
using Mat4f = Mat<float, 4, 4>;
using Mat2d = Mat<double, 2, 2>;
using Mat3d = Mat<double, 3, 3>;
using Mat4d = Mat<double, 4, 4>;

} // namespace ema

#include "func.hpp"
