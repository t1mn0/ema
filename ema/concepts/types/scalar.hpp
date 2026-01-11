#pragma once

#include <concepts>
#include <type_traits>

namespace ema::types {

template <typename T>
concept Scalar =
    std::is_arithmetic_v<T> &&
    !std::same_as<T, bool> &&
    !std::same_as<T, char> &&
    !std::same_as<T, char16_t> &&
    !std::same_as<T, char32_t> &&
    !std::same_as<T, wchar_t> &&
    std::is_trivially_copyable_v<T>;

} // namespace ema::types
