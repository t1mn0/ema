#pragma once

#include "ema/vec/vec.hpp"

namespace ema {

template <types::Scalar T>
class BiVec3 {
  public:
    T xy, yz, zx;

    constexpr BiVec3() : xy(0), yz(0), zx(0) {}
    constexpr BiVec3(T xy, T yz, T zx) : xy(xy), yz(yz), zx(zx) {}

    constexpr BiVec3& operator*=(T scale) {
        xy *= scale;
        yz *= scale;
        zx *= scale;
        return *this;
    }

    constexpr BiVec3 operator-() const { return {-xy, -yz, -zx}; }
};

template <types::Scalar T>
constexpr BiVec3<T> wedge(const Vec<T, 3>& a, const Vec<T, 3>& b) {
    return {
        a.x() * b.y() - a.y() * b.x(), // XY
        a.y() * b.z() - a.z() * b.y(), // YZ
        a.z() * b.x() - a.x() * b.z()  // ZX
    };
}

} // namespace ema
