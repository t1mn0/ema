#pragma once

#include <cstddef> // for: size_t
#include <iterator>

#include "ema/concepts/types/scalar.hpp"

namespace ema {

// `alignas()` for SIMD optimization through data alignment and padding;
// Alignment can also improve caching in some cases (for example, escape false-sharing);
// It aligns the class (not the `components` inner array)
template <types::Scalar T, size_t N> requires(N > 0 && N <= 4)
class alignas(sizeof(T) * (N == 3 ? 4 : N)) Vec {
  private: // member-fields:
    T components[N];

  public: // member-functions:
    // default constructor:
    constexpr Vec() : components{T(0)} {} // returning zero-vector;

    // constructor by num: each element of the vector being created is equal to num:
    constexpr Vec(T num) {
        for (int i = 0; i < N; ++i) {
            components[i] = num;
        }
    }

    constexpr static Vec<T, N> Unit(size_t axis = 0) {
        // vec is Vec<T,N>::Unit <=> len(vec) == 1;
        Vec<T, N> unit;
        // for an N-dimensional vector, it is possible to choose the direction along the axis
        // the component of the unit vector is maximal (i.e. =1):
        // Vec<T, 3>::Unit(0) is {1, 0, 0}
        // Vec<T, 3>::Unit(1) is {0, 1, 0}
        // Vec<T, 3>::Unit(2) is {0, 0, 1}
        unit[axis % N] = 1;
        return unit;
    }

    // zero by addition vector:
    constexpr static Vec<T, N> Zero() {
        // v1 : Vec<T,N>        ---+
        //                         |---> v1 + v2 = v1 = v2 + v1;
        // v2 : Vec<T,N>::Zero  ---+
        //
        // v1 + inverse(v1) = Vec<T,N>::Zero;
        // note: inverse is:
        // v1 +       -(v1) = Vec<T,N>::Zero;
        //
        // => Vec<T,2>::Zero is {0,0}
        //    Vec<T,3>::Zero is {0,0,0}
        //    Vec<T,4>::Zero is {0,0,0,0}

        return Vec<T, N>{};
    }

    // basic general constructor:
    // allows to pass enumerations of specific values of each component:
    // Vec<T,3>(1,4,8) is {1, 4, 8}
    template <typename... Args>
    constexpr Vec(Args... args) : components{static_cast<T>(args)...} {
        static_assert(sizeof...(Args) == N, "Vec: invalid number of arguments");
    }

    constexpr size_t dimension() const { return N; }

    // clang-format off
    // Member functions for iterating over vector components in forward/reverse order:
    constexpr       T* begin()       noexcept { return components; }
    constexpr const T* begin() const noexcept { return components; }
    constexpr       T* end()         noexcept { return components + N; }
    constexpr const T* end()   const noexcept { return components + N; }

    constexpr const T* cbegin() const noexcept { return components; }
    constexpr const T* cend()   const noexcept { return components + N; }

    constexpr std::reverse_iterator<T*> rbegin()             noexcept { return std::reverse_iterator<T*>(end()); }
    constexpr std::reverse_iterator<const T*> rbegin() const noexcept { return std::reverse_iterator<const T*>(end()); }
    constexpr std::reverse_iterator<T*> rend()               noexcept { return std::reverse_iterator<T*>(begin()); }
    constexpr std::reverse_iterator<const T*> rend() const   noexcept { return std::reverse_iterator<const T*>(begin()); }
    // clang-format on

    // component-by-component subtraction:
    Vec operator-() const {
        Vec<T, N> result;
        for (size_t i = 0; i < N; ++i) {
            result.components[i] = -components[i];
        }
        return result;
    }

    // component-by-component addition with assignment:
    Vec& operator+=(const Vec& other) {
        for (size_t i = 0; i < N; ++i) {
            components[i] += other.components[i];
        }
        return *this;
    }

    // component-by-component addition with subtraction:
    Vec& operator-=(const Vec& other) {
        for (size_t i = 0; i < N; ++i) {
            components[i] -= other.components[i];
        }
        return *this;
    }

    // multiplying each component by scalar with the assignment:
    Vec& operator*=(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            components[i] *= scalar;
        }
        return *this;
    }

    // division each component by scalar with the assignment:
    Vec& operator/=(T scalar) {
        for (size_t i = 0; i < N; ++i) {
            components[i] /= scalar;
        }
        return *this;
    }

    // clang-format off
    // access operator for components:
    constexpr       T& operator[](size_t i)       { return components[i]; }
    constexpr const T& operator[](size_t i) const { return components[i]; }

    // access member functions for components considering the dimensions:
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

// Free binary arithmetic operators:
template <types::Scalar T, size_t N>
constexpr Vec<T, N> operator+(Vec<T, N> lhs, const Vec<T, N>& rhs) {
    lhs += rhs;
    return lhs;
}

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
constexpr Vec<T, N> operator/(Vec<T, N> vec, T scalar) {
    vec /= scalar;
    return vec;
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
