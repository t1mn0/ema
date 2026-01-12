#pragma once

#include "ema/vec/vec.hpp"

namespace ema {

template <types::Scalar T>
class BiVec3 {
  private:        // member fields:
    T xy_ = T(0); // coefficient for the e₁∧e₂ (xy-plane) basis bivector
    T yz_ = T(0); // coefficient for the e₂∧e₃ (yz-plane) basis bivector
    T zx_ = T(0); // coefficient for the e₃∧e₁ (zx-plane) basis bivector

  public: // member functions:
    constexpr BiVec3() : xy_(0), yz_(0), zx_(0) {}
    constexpr BiVec3(T xy, T yz, T zx) : xy_(xy), yz_(yz), zx_(zx) {}

    // clang-format off
    // element access member-functions:
    constexpr const T& xy() const noexcept { return xy_; }
    constexpr       T& xy()       noexcept { return xy_; }
    constexpr const T& yz() const noexcept { return yz_; }
    constexpr       T& yz()       noexcept { return yz_; }
    constexpr const T& zx() const noexcept { return zx_; }
    constexpr       T& zx()       noexcept { return zx_; }
    // clang-format on

    // component-by-component addition with assignment:
    BiVec3& operator+=(const BiVec3& oth) {
        xy_ += oth.xy_;
        yz_ += oth.yz_;
        zx_ += oth.zx_;
        return *this;
    }

    // component-by-component subtraction with assignment:
    BiVec3& operator-=(const BiVec3& oth) {
        xy_ -= oth.xy_;
        yz_ -= oth.yz_;
        zx_ -= oth.zx_;
        return *this;
    }

    // multiplying each component by scalar with the assignment:
    constexpr BiVec3& operator*=(T scale) {
        xy_ *= scale;
        yz_ *= scale;
        zx_ *= scale;
        return *this;
    }

    constexpr BiVec3 operator-() const { return {-xy_, -yz_, -zx_}; }
};

// Free binary arithmetic operators:
template <types::Scalar T>
constexpr BiVec3<T> operator+(BiVec3<T> lhs, const BiVec3<T>& rhs) {
    lhs += rhs;
    return lhs;
}

template <types::Scalar T>
constexpr BiVec3<T> operator-(BiVec3<T> lhs, const BiVec3<T>& rhs) {
    lhs -= rhs;
    return lhs;
}

template <types::Scalar T>
constexpr BiVec3<T> operator*(BiVec3<T> bivec, T scalar) {
    bivec *= scalar;
    return bivec;
}

template <types::Scalar T>
constexpr BiVec3<T> operator*(T scalar, BiVec3<T> bivec) {
    return bivec * scalar;
}

template <types::Scalar T>
constexpr BiVec3<T> wedge(const Vec<T, 3>& a, const Vec<T, 3>& b) {
    return {
        a.x() * b.y() - a.y() * b.x(), // XY-plane component
        a.y() * b.z() - a.z() * b.y(), // YZ-plane component
        a.z() * b.x() - a.x() * b.z()  // ZX-plane component
    };
}

} // namespace ema
