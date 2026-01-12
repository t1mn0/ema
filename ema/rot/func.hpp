#pragma once

#include "ema/mat/mat.hpp"
#include "rot3.hpp"

namespace ema::make {

template <types::Scalar T>
constexpr Rotor3<T> reverse(const Rotor3<T>& rot) {
    return {rot.scalar(), -rot.bivector()};
}

template <types::Scalar T>
constexpr Mat<T, 3, 3> mat_from_rot(const Rotor3<T>& rot) {
    Vec<T, 3> e1 = rot.rotate({1, 0, 0});
    Vec<T, 3> e2 = rot.rotate({0, 1, 0});
    Vec<T, 3> e3 = rot.rotate({0, 0, 1});
    return Mat<T, 3, 3>{e1, e2, e3};
}

} // namespace ema::make
