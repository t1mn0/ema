#pragma once

#include "angle.hpp"

namespace ema::constants {

template <types::Scalar T>
constexpr Angle<T> zero_angle = Angle<T>::rad(T(0));

template <types::Scalar T>
constexpr Angle<T> right_angle = Angle<T>::deg(T(90));

template <types::Scalar T>
constexpr Angle<T> straight_angle = Angle<T>::deg(T(180));

template <types::Scalar T>
constexpr Angle<T> full_angle = Angle<T>::deg(T(360));

template <types::Scalar T>
constexpr Angle<T> fov_60 = Angle<T>::deg(T(60));

template <types::Scalar T>
constexpr Angle<T> fov_90 = Angle<T>::deg(T(90));

template <types::Scalar T>
constexpr Angle<T> fov_120 = Angle<T>::deg(T(120));

} // namespace ema::constants
