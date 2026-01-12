#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "ema/vec/vec.hpp"

template <typename T>
class VecFuncTest : public ::testing::Test {
  protected:
    using Vec2 = ema::Vec<T, 2>;
    using Vec3 = ema::Vec<T, 3>;
    using Vec4 = ema::Vec<T, 4>;

    const T zero = static_cast<T>(0);
    const T one = static_cast<T>(1);
    const T two = static_cast<T>(2);
    const T three = static_cast<T>(3);
    const T epsilon = std::numeric_limits<T>::epsilon();

    bool approx_equal(T a, T b, T tol = std::numeric_limits<T>::epsilon() * 100) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(a - b) <= tol;
        } else {
            return a == b;
        }
    }
};

// functions over Vec<int, N> can sometimes return an inaccurate value due to the integer limitation
// so there are testing only float/double
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(VecFuncTest, TestTypes);

TYPED_TEST(VecFuncTest, Length) {
    typename TestFixture::Vec2 zero2 = TestFixture::Vec2::Zero();
    EXPECT_EQ(ema::len(zero2), this->zero);

    typename TestFixture::Vec3 zero3 = TestFixture::Vec3::Zero();
    EXPECT_EQ(ema::len(zero3), this->zero);

    typename TestFixture::Vec2 unit2{1, 0};
    EXPECT_TRUE(this->approx_equal(ema::len(unit2), this->one));

    typename TestFixture::Vec3 unit3{0, 1, 0};
    EXPECT_TRUE(this->approx_equal(ema::len(unit3), this->one));

    typename TestFixture::Vec3 vec3{3, 4, 0};
    EXPECT_TRUE(this->approx_equal(ema::len(vec3), static_cast<TypeParam>(5)));

    typename TestFixture::Vec2 vec2{-3, -4};
    EXPECT_TRUE(this->approx_equal(ema::len(vec2), static_cast<TypeParam>(5)));
}

TYPED_TEST(VecFuncTest, LengthSquared) {
    typename TestFixture::Vec3 vec{this->two, this->three, this->two};
    TypeParam expected = this->two * this->two + this->three * this->three + this->two * this->two;
    EXPECT_EQ(ema::len_squared(vec), expected);

    typename TestFixture::Vec3 zero = TestFixture::Vec3::Zero();
    EXPECT_EQ(ema::len_squared(zero), this->zero);
}

TYPED_TEST(VecFuncTest, Normalize) {
    typename TestFixture::Vec3 unit_x = TestFixture::Vec3::Unit(0);
    typename TestFixture::Vec3 norm1 = ema::normalize(unit_x);
    EXPECT_TRUE(this->approx_equal(ema::len(norm1), this->one));
    EXPECT_TRUE(this->approx_equal(norm1.x(), this->one));
    EXPECT_TRUE(this->approx_equal(norm1.y(), this->zero));
    EXPECT_TRUE(this->approx_equal(norm1.z(), this->zero));

    typename TestFixture::Vec3 vec{2, 0, 0};
    typename TestFixture::Vec3 norm2 = ema::normalize(vec);
    EXPECT_TRUE(this->approx_equal(ema::len(norm2), this->one));
    EXPECT_TRUE(this->approx_equal(norm2.x(), this->one));
    EXPECT_TRUE(this->approx_equal(norm2.y(), this->zero));
    EXPECT_TRUE(this->approx_equal(norm2.z(), this->zero));

    typename TestFixture::Vec3 vec3{1, 1, 0};
    typename TestFixture::Vec3 norm3 = ema::normalize(vec3);
    EXPECT_TRUE(this->approx_equal(ema::len(norm3), this->one));
    TypeParam expected_val = static_cast<TypeParam>(1.0 / std::sqrt(2.0));
    EXPECT_TRUE(this->approx_equal(norm3.x(), expected_val));
    EXPECT_TRUE(this->approx_equal(norm3.y(), expected_val));
    EXPECT_TRUE(this->approx_equal(norm3.z(), this->zero));

    typename TestFixture::Vec3 vec4{-3, 0, 0};
    typename TestFixture::Vec3 norm4 = ema::normalize(vec4);
    EXPECT_TRUE(this->approx_equal(ema::len(norm4), this->one));
    EXPECT_TRUE(this->approx_equal(norm4.x(), -this->one));
    EXPECT_TRUE(this->approx_equal(norm4.y(), this->zero));
    EXPECT_TRUE(this->approx_equal(norm4.z(), this->zero));
}

TYPED_TEST(VecFuncTest, DotProduct) {
    typename TestFixture::Vec3 v1{1, 0, 0};
    typename TestFixture::Vec3 v2{0, 1, 0};
    EXPECT_EQ(ema::dot(v1, v2), this->zero);

    typename TestFixture::Vec3 v3{2, 0, 0};
    typename TestFixture::Vec3 v4{3, 0, 0};
    EXPECT_EQ(ema::dot(v3, v4), this->two * this->three);

    typename TestFixture::Vec3 a{1, 2, 3};
    typename TestFixture::Vec3 b{4, 5, 6};
    TypeParam expected = 1 * 4 + 2 * 5 + 3 * 6;
    EXPECT_EQ(ema::dot(a, b), expected);
    // comutativity:
    EXPECT_EQ(ema::dot(a, b), ema::dot(b, a));

    // distributivity:
    typename TestFixture::Vec3 c{7, 8, 9};
    TypeParam left = ema::dot(a + b, c);
    TypeParam right = ema::dot(a, c) + ema::dot(b, c);
    EXPECT_TRUE(this->approx_equal(left, right));
}

TYPED_TEST(VecFuncTest, Distance) {
    typename TestFixture::Vec3 a{1, 2, 3};
    typename TestFixture::Vec3 b{4, 6, 3};

    EXPECT_EQ(ema::distance(a, a), this->zero);
    EXPECT_EQ(ema::distance_squared(a, a), this->zero);

    TypeParam dx = 4 - 1;
    TypeParam dy = 6 - 2;
    TypeParam dz = 3 - 3;
    TypeParam expected_squared = dx * dx + dy * dy + dz * dz;
    TypeParam expected = std::sqrt(expected_squared);

    EXPECT_TRUE(this->approx_equal(ema::distance(a, b), expected));
    EXPECT_TRUE(this->approx_equal(ema::distance_squared(a, b), expected_squared));

    // comutativity:
    EXPECT_TRUE(this->approx_equal(ema::distance(a, b), ema::distance(b, a)));
    EXPECT_TRUE(this->approx_equal(ema::distance_squared(a, b), ema::distance_squared(b, a)));
}

TYPED_TEST(VecFuncTest, CrossProduct) {
    typename TestFixture::Vec3 i{1, 0, 0};
    typename TestFixture::Vec3 j{0, 1, 0};
    typename TestFixture::Vec3 k{0, 0, 1};

    // i × j = k
    typename TestFixture::Vec3 result1 = ema::cross_product(i, j);
    EXPECT_TRUE(this->approx_equal(result1.x(), this->zero));
    EXPECT_TRUE(this->approx_equal(result1.y(), this->zero));
    EXPECT_TRUE(this->approx_equal(result1.z(), this->one));

    // j × k = i
    typename TestFixture::Vec3 result2 = ema::cross_product(j, k);
    EXPECT_TRUE(this->approx_equal(result2.x(), this->one));
    EXPECT_TRUE(this->approx_equal(result2.y(), this->zero));
    EXPECT_TRUE(this->approx_equal(result2.z(), this->zero));

    // k × i = j
    typename TestFixture::Vec3 result3 = ema::cross_product(k, i);
    EXPECT_TRUE(this->approx_equal(result3.x(), this->zero));
    EXPECT_TRUE(this->approx_equal(result3.y(), this->one));
    EXPECT_TRUE(this->approx_equal(result3.z(), this->zero));

    // acomutativity: a × b = -(b × a)
    typename TestFixture::Vec3 a{1, 2, 3};
    typename TestFixture::Vec3 b{4, 5, 6};
    typename TestFixture::Vec3 cross_ab = ema::cross_product(a, b);
    typename TestFixture::Vec3 cross_ba = ema::cross_product(b, a);
    EXPECT_TRUE(this->approx_equal(cross_ab.x(), -cross_ba.x()));
    EXPECT_TRUE(this->approx_equal(cross_ab.y(), -cross_ba.y()));
    EXPECT_TRUE(this->approx_equal(cross_ab.z(), -cross_ba.z()));

    // orthogonality of the result to the original vectors
    EXPECT_TRUE(this->approx_equal(ema::dot(cross_ab, a), this->zero));
    EXPECT_TRUE(this->approx_equal(ema::dot(cross_ab, b), this->zero));
}

TYPED_TEST(VecFuncTest, TripleProduct) {
    typename TestFixture::Vec3 i{1, 0, 0};
    typename TestFixture::Vec3 j{0, 1, 0};
    typename TestFixture::Vec3 k{0, 0, 1};

    // [i, j, k] = 1
    TypeParam result1 = ema::triple_product(i, j, k);
    EXPECT_TRUE(this->approx_equal(result1, this->one));

    // [j, i, k] = -1 (acomutativity)
    TypeParam result2 = ema::triple_product(j, i, k);
    EXPECT_TRUE(this->approx_equal(result2, -this->one));

    typename TestFixture::Vec3 a{1, 2, 3};
    typename TestFixture::Vec3 b{4, 5, 6};
    typename TestFixture::Vec3 c{7, 8, 9};

    TypeParam t1 = ema::triple_product(a, b, c);
    TypeParam t2 = ema::triple_product(b, c, a);
    TypeParam t3 = ema::triple_product(c, a, b);

    EXPECT_TRUE(this->approx_equal(t1, t2));
    EXPECT_TRUE(this->approx_equal(t2, t3));

    TypeParam t4 = ema::triple_product(b, a, c);
    EXPECT_TRUE(this->approx_equal(t1, -t4));
}

TYPED_TEST(VecFuncTest, Reflect) {
    typename TestFixture::Vec3 incident{1, -1, 0};
    typename TestFixture::Vec3 normal{0, 1, 0};

    typename TestFixture::Vec3 reflected = ema::reflect(incident, normal);

    // expected: (1, 1, 0)
    EXPECT_TRUE(this->approx_equal(reflected.x(), this->one));
    EXPECT_TRUE(this->approx_equal(reflected.y(), this->one));
    EXPECT_TRUE(this->approx_equal(reflected.z(), this->zero));

    // incidence_angle == reflection_angle
    auto angle_incident = ema::angle(incident, -normal);
    auto angle_reflected = ema::angle(reflected, normal);
    EXPECT_TRUE(this->approx_equal(angle_incident.as_rad(), angle_reflected.as_rad()));

    EXPECT_TRUE(this->approx_equal(ema::len(incident), ema::len(reflected)));

    typename TestFixture::Vec3 incident2{0, -1, 0};
    typename TestFixture::Vec3 normal2{1, 1, 0};
    typename TestFixture::Vec3 norm_normal2 = ema::normalize(normal2);
    typename TestFixture::Vec3 reflected2 = ema::reflect(incident2, norm_normal2);

    EXPECT_TRUE(this->approx_equal(ema::len(incident2), ema::len(reflected2)));
}

TYPED_TEST(VecFuncTest, ProjectReject) {
    typename TestFixture::Vec3 a{2, 3, 4};
    typename TestFixture::Vec3 b{1, 0, 0};

    typename TestFixture::Vec3 proj = project(a, b);
    typename TestFixture::Vec3 rej = reject(a, b);

    EXPECT_TRUE(this->approx_equal(cross_product(proj, b).x(), this->zero));
    EXPECT_TRUE(this->approx_equal(cross_product(proj, b).y(), this->zero));
    EXPECT_TRUE(this->approx_equal(cross_product(proj, b).z(), this->zero));

    EXPECT_TRUE(this->approx_equal(ema::dot(rej, b), this->zero));

    typename TestFixture::Vec3 sum = proj + rej;
    EXPECT_TRUE(this->approx_equal(sum.x(), a.x()));
    EXPECT_TRUE(this->approx_equal(sum.y(), a.y()));
    EXPECT_TRUE(this->approx_equal(sum.z(), a.z()));

    typename TestFixture::Vec3 proj_self = ema::project(a, a);
    EXPECT_TRUE(this->approx_equal(ema::len(proj_self), ema::len(a)));

    typename TestFixture::Vec3 zero = TestFixture::Vec3::Zero();
    typename TestFixture::Vec3 proj_zero = ema::project(a, zero);
    EXPECT_TRUE(ema::is_zero(proj_zero));
}

TYPED_TEST(VecFuncTest, Lerp) {
    typename TestFixture::Vec3 a{1, 2, 3};
    typename TestFixture::Vec3 b{4, 5, 6};

    // t = 0 -> a
    typename TestFixture::Vec3 result0 = ema::lerp(a, b, this->zero);
    EXPECT_TRUE(this->approx_equal(result0.x(), a.x()));
    EXPECT_TRUE(this->approx_equal(result0.y(), a.y()));
    EXPECT_TRUE(this->approx_equal(result0.z(), a.z()));

    // t = 1 -> b
    typename TestFixture::Vec3 result1 = ema::lerp(a, b, this->one);
    EXPECT_TRUE(this->approx_equal(result1.x(), b.x()));
    EXPECT_TRUE(this->approx_equal(result1.y(), b.y()));
    EXPECT_TRUE(this->approx_equal(result1.z(), b.z()));

    // t = 0.5
    typename TestFixture::Vec3 result_half = ema::lerp(a, b, static_cast<TypeParam>(0.5));
    EXPECT_TRUE(this->approx_equal(result_half.x(), static_cast<TypeParam>(2.5)));
    EXPECT_TRUE(this->approx_equal(result_half.y(), static_cast<TypeParam>(3.5)));
    EXPECT_TRUE(this->approx_equal(result_half.z(), static_cast<TypeParam>(4.5)));

    // t out [0, 1] -> extrapolation
    typename TestFixture::Vec3 result_neg = ema::lerp(a, b, static_cast<TypeParam>(-0.5));
    typename TestFixture::Vec3 result_pos = ema::lerp(a, b, static_cast<TypeParam>(1.5));

    EXPECT_TRUE(this->approx_equal(
        ema::lerp(a, b, static_cast<TypeParam>(0.3)).x(),
        a.x() + static_cast<TypeParam>(0.3) * (b.x() - a.x())));
}

TYPED_TEST(VecFuncTest, ClampLength) {
    typename TestFixture::Vec3 vec{3, 4, 0}; // len = 5

    typename TestFixture::Vec3 result1 = ema::clamp_len(vec, static_cast<TypeParam>(10));
    EXPECT_TRUE(this->approx_equal(result1.x(), vec.x()));
    EXPECT_TRUE(this->approx_equal(result1.y(), vec.y()));
    EXPECT_TRUE(this->approx_equal(result1.z(), vec.z()));

    TypeParam max_len = static_cast<TypeParam>(2.5);
    typename TestFixture::Vec3 result2 = ema::clamp_len(vec, max_len);
    EXPECT_TRUE(this->approx_equal(len(result2), max_len));

    // direction preserved:
    typename TestFixture::Vec3 norm_vec = ema::normalize(vec);
    typename TestFixture::Vec3 norm_result = ema::normalize(result2);
    EXPECT_TRUE(this->approx_equal(norm_vec.x(), norm_result.x()));
    EXPECT_TRUE(this->approx_equal(norm_vec.y(), norm_result.y()));
    EXPECT_TRUE(this->approx_equal(norm_vec.z(), norm_result.z()));

    // zero vec:
    typename TestFixture::Vec3 zero = TestFixture::Vec3::Zero();
    typename TestFixture::Vec3 result3 = ema::clamp_len(zero, max_len);
    EXPECT_TRUE(is_zero(result3));
}

TYPED_TEST(VecFuncTest, Abs) {
    typename TestFixture::Vec3 vec{-1, 2, -3};
    typename TestFixture::Vec3 result = ema::abs(vec);

    EXPECT_EQ(result.x(), this->one);
    EXPECT_EQ(result.y(), this->two);
    EXPECT_EQ(result.z(), this->three);

    typename TestFixture::Vec3 pos_vec(1, 2, 3);
    typename TestFixture::Vec3 pos_result = ema::abs(pos_vec);
    EXPECT_TRUE(this->approx_equal(pos_result.x(), pos_vec.x()));
    EXPECT_TRUE(this->approx_equal(pos_result.y(), pos_vec.y()));
    EXPECT_TRUE(this->approx_equal(pos_result.z(), pos_vec.z()));

    typename TestFixture::Vec3 zero = TestFixture::Vec3::Zero();
    typename TestFixture::Vec3 zero_result = ema::abs(zero);
    EXPECT_TRUE(is_zero(zero_result));
}

TYPED_TEST(VecFuncTest, Angle) {
    // 0:
    typename TestFixture::Vec3 a1{1, 0, 0};
    typename TestFixture::Vec3 b1{2, 0, 0};
    EXPECT_TRUE(this->approx_equal(ema::angle(a1, b1).as_rad(), this->zero));

    // π/2:
    typename TestFixture::Vec3 a2{1, 0, 0};
    typename TestFixture::Vec3 b2{0, 1, 0};
    TypeParam expected_90 = static_cast<TypeParam>(M_PI / 2.0);
    EXPECT_TRUE(this->approx_equal(ema::angle(a2, b2).as_rad(), expected_90));

    // π:
    typename TestFixture::Vec3 a3{1, 0, 0};
    typename TestFixture::Vec3 b3{-1, 0, 0};
    TypeParam expected_180 = static_cast<TypeParam>(M_PI);
    EXPECT_TRUE(this->approx_equal(ema::angle(a3, b3).as_rad(), expected_180));

    // random:
    typename TestFixture::Vec3 a4{1, 1, 0};
    typename TestFixture::Vec3 b4{1, 0, 0};
    auto angle_45 = ema::angle(a4, b4);
    TypeParam expected_45 = static_cast<TypeParam>(M_PI / 4.0);
    EXPECT_TRUE(this->approx_equal(angle_45.as_rad(), expected_45));

    // symmetry:
    EXPECT_TRUE(this->approx_equal(ema::angle(a4, b4).as_rad(), ema::angle(b4, a4).as_rad()));

    // angle with zero-vec should return 0:
    typename TestFixture::Vec3 zero = TestFixture::Vec3::Zero();
    EXPECT_TRUE(this->approx_equal(ema::angle(a4, zero).as_rad(), this->zero));
}

TYPED_TEST(VecFuncTest, IsOrthogonal) {
    typename TestFixture::Vec3 a{1, 0, 0};
    typename TestFixture::Vec3 b{0, 1, 0};
    EXPECT_TRUE(ema::is_orthogonal(a, b));

    typename TestFixture::Vec3 c{1, 1, 0};
    typename TestFixture::Vec3 d{1, 0, 0};
    EXPECT_FALSE(ema::is_orthogonal(c, d));

    EXPECT_FALSE(ema::is_orthogonal(a, a));

    typename TestFixture::Vec3 zero = TestFixture::Vec3::Zero();
    EXPECT_TRUE(ema::is_orthogonal(a, zero));
    EXPECT_TRUE(ema::is_orthogonal(zero, a));

    if constexpr (std::is_floating_point_v<TypeParam>) {
        typename TestFixture::Vec3 e{1, this->epsilon / 2, 0};
        typename TestFixture::Vec3 f{0, 1, 0};
        EXPECT_TRUE(ema::is_orthogonal(e, f, this->epsilon));
    }
}

TYPED_TEST(VecFuncTest, CombinedOperations) {
    // |a|² = a*a
    typename TestFixture::Vec3 a{2, 3, 4};
    EXPECT_TRUE(this->approx_equal(ema::len_squared(a), ema::dot(a, a)));

    // |a × b|² = |a|²|b|² - (a·b)²
    typename TestFixture::Vec3 b{5, 6, 7};
    TypeParam left = ema::len_squared(ema::cross_product(a, b));
    TypeParam right = ema::len_squared(a) * ema::len_squared(b) - ema::dot(a, b) * ema::dot(a, b);
    EXPECT_TRUE(this->approx_equal(left, right));

    // |normalize(v)| = 1 (v is not zero-vec)
    typename TestFixture::Vec3 c{3, 4, 5};
    EXPECT_TRUE(this->approx_equal(ema::len(ema::normalize(c)), this->one));

    // |a + b| ≤ |a| + |b|
    typename TestFixture::Vec3 d{1, 2, 3};
    typename TestFixture::Vec3 e{4, 5, 6};
    EXPECT_LE(ema::len(d + e), ema::len(d) + ema::len(e));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
