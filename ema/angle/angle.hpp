#pragma once

#include <compare> // for: operator<=>
#include <numbers> // for: pi_v

#include "ema/concepts/types/scalar.hpp"

namespace ema {

// User can increase the accuracy of calculating angles if necessary:
template <types::Scalar T = float>
class Angle {
  public:
    T rad_val;

    constexpr Angle() : rad_val(0) {}
    explicit constexpr Angle(T r) : rad_val(r) {}

    static constexpr Angle rad(T r) { return Angle(r); }
    static constexpr Angle deg(T d) {
        return Angle(d * (std::numbers::pi_v<T> / T(180)));
    }

    constexpr T as_deg() const {
        return rad_val * (T(180) / std::numbers::pi_v<T>);
    }

    constexpr T as_rad() const { return rad_val; }

    constexpr auto operator<=>(const Angle<T>&) const = default;
    constexpr Angle operator+(const Angle& oth) const { return Angle(rad_val + oth.rad_val); }
    constexpr Angle operator-(const Angle& oth) const { return Angle(rad_val - oth.rad_val); }

    constexpr Angle& operator-=(const Angle& oth) {
        rad_val -= oth.rad_val;
        return *this;
    }

    constexpr Angle& operator+=(const Angle& oth) {
        rad_val += oth.rad_val;
        return *this;
    }
};

// literals:
//      Angle a = 90_deg;
//      Angle b = 120_rad;
namespace literals {
constexpr Angle<float> operator""_deg(long double d) {
    return Angle<float>::deg(static_cast<float>(d));
}
constexpr Angle<float> operator""_rad(long double r) {
    return Angle<float>::rad(static_cast<float>(r));
}
} // namespace literals

} // namespace ema
