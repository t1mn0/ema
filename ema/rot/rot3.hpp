#pragma once

#include "ema/bivec/bivec3.hpp"
#include "ema/mat/mat.hpp"

namespace ema {

template <types::Scalar T>
class Rotor3 {
  public:
    T s;
    BiVec3<T> bv;

    constexpr Rotor3() : s(1), bv() {}
    constexpr Rotor3(T s, const BiVec3<T>& b) : s(s), bv(b) {}

    static Rotor3 from_angle_plane(T angle_rad, const BiVec3<T>& plane) {
        T half_angle = angle_rad / T(2);
        T sine = std::sin(half_angle);
        return {std::cos(half_angle), BiVec3<T>(plane.xy * sine, plane.yz * sine, plane.zx * sine)};
    }

    Rotor3 operator*(const Rotor3& other) const {
        return {
            s * other.s - (bv.xy * other.bv.xy + bv.yz * other.bv.yz + bv.zx * other.bv.zx),
            BiVec3<T>{
                s * other.bv.xy + bv.xy * other.s + bv.zx * other.bv.yz - bv.yz * other.bv.zx,
                s * other.bv.yz + bv.yz * other.s + bv.xy * other.bv.zx - bv.zx * other.bv.xy,
                s * other.bv.zx + bv.zx * other.s + bv.yz * other.bv.xy - bv.xy * other.bv.yz}};
    }

    constexpr Vec<T, 3> rotate(const Vec<T, 3>& v) const {
        // sandwich product: v' = R * v * R_rev
        // w = R * v (geometric product: [rot, v])
        T wx = s * v.x() + bv.xy * v.y() - bv.zx * v.z();
        T wy = s * v.y() - bv.xy * v.x() + bv.yz * v.z();
        T wz = s * v.z() + bv.zx * v.x() - bv.yz * v.y();
        T tri = bv.xy * v.z() + bv.yz * v.x() + bv.zx * v.y();

        return {
            wx * s + wy * bv.xy - wz * bv.zx + tri * bv.yz,
            wy * s - wx * bv.xy + wz * bv.yz + tri * bv.zx,
            wz * s + wx * bv.zx - wy * bv.yz + tri * bv.xy};
    }

    constexpr Mat<T, 3, 3> to_matrix() const {
        Vec<T, 3> e1 = rotate({1, 0, 0});
        Vec<T, 3> e2 = rotate({0, 1, 0});
        Vec<T, 3> e3 = rotate({0, 0, 1});
        return Mat<T, 3, 3>{e1, e2, e3};
    }
};

} // namespace ema
