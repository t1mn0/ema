#pragma once

#include "ema/bivec/bivec3.hpp"

namespace ema {

template <types::Scalar T>
class Rotor3 {
  private: // member-fields:
    T s;
    BiVec3<T> bv;

  public: // member-functions:
    constexpr Rotor3() : s(static_cast<T>(1)), bv() {}
    constexpr Rotor3(T s, const BiVec3<T>& b) : s(s), bv(b) {}

    constexpr static Rotor3 Identity() {
        return {T(1), BiVec3<T>{0, 0, 0}};
    }

    static Rotor3 from_angle_plane(Angle<T> angle, const BiVec3<T>& plane) {
        T half_angle = angle.as_rad() / T(2);
        T cos_half = std::cos(half_angle);
        T sin_half = std::sin(half_angle);

        T norm = std::sqrt(plane.xy() * plane.xy() +
                           plane.yz() * plane.yz() +
                           plane.zx() * plane.zx());

        if (norm > std::numeric_limits<T>::epsilon()) {
            T inv_norm = T(1) / norm;
            return {
                cos_half,
                BiVec3<T>{
                    plane.xy() * sin_half * inv_norm,
                    plane.yz() * sin_half * inv_norm,
                    plane.zx() * sin_half * inv_norm}};
        }

        return Identity();
    }

    static Rotor3 from_axis_angle(const Vec<T, 3>& axis, Angle<T> angle) {
        T half_angle = angle.as_rad() / T(2);
        T sin_half = std::sin(half_angle);
        Vec<T, 3> norm_axis = ema::normalize(axis);

        return {
            std::cos(half_angle),
            BiVec3<T>{
                norm_axis.z() * sin_half,
                norm_axis.x() * sin_half,
                norm_axis.y() * sin_half}};
    }

    constexpr const T& scalar() const noexcept { return s; }
    constexpr T& scalar() noexcept { return s; }
    constexpr const BiVec3<T>& bivector() const noexcept { return bv; }
    constexpr BiVec3<T>& bivector() noexcept { return bv; }

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
};

} // namespace ema

#include "func.hpp"
