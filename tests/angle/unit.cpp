#include <cmath>
#include <limits>
#include <numbers>

#include <gtest/gtest.h>

#include "ema/angle/angle.hpp"
#include "ema/angle/const.hpp"

template <typename T>
class AngleTest : public ::testing::Test {
  protected:
    const T epsilon = std::numeric_limits<T>::epsilon() * 100;
    const T pi = std::numbers::pi_v<T>;

    bool approx_equal(T a, T b) {
        return std::abs(a - b) <= epsilon;
    }

    bool approx_equal(const ema::Angle<T>& a, const ema::Angle<T>& b) {
        return std::abs(a.as_rad() - b.as_rad()) <= epsilon;
    }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(AngleTest, TestTypes);

TYPED_TEST(AngleTest, DefaultConstructor) {
    ema::Angle<TypeParam> angle;
    EXPECT_TRUE(this->approx_equal(angle.as_rad(), 0.0));
    EXPECT_TRUE(this->approx_equal(angle.as_deg(), 0.0));
}

TYPED_TEST(AngleTest, RadConstructor) {
    ema::Angle<TypeParam> angle(this->pi);
    EXPECT_TRUE(this->approx_equal(angle.as_rad(), this->pi));
    EXPECT_TRUE(this->approx_equal(angle.as_deg(), 180));
}

TYPED_TEST(AngleTest, RadFactoryMethod) {
    auto angle = ema::Angle<TypeParam>::rad(this->pi / 2);
    EXPECT_TRUE(this->approx_equal(angle.as_rad(), this->pi / 2));
    EXPECT_TRUE(this->approx_equal(angle.as_deg(), 90));
}

TYPED_TEST(AngleTest, DegFactoryMethod) {
    auto angle = ema::Angle<TypeParam>::deg(45);
    EXPECT_TRUE(this->approx_equal(angle.as_rad(), this->pi / 4));
    EXPECT_TRUE(this->approx_equal(angle.as_deg(), 45));
}

TYPED_TEST(AngleTest, NegativeAngles) {
    auto neg_rad = ema::Angle<TypeParam>::rad(-this->pi);
    EXPECT_TRUE(this->approx_equal(neg_rad.as_rad(), -this->pi));
    EXPECT_TRUE(this->approx_equal(neg_rad.as_deg(), -180));

    auto neg_deg = ema::Angle<TypeParam>::deg(-90);
    EXPECT_TRUE(this->approx_equal(neg_deg.as_rad(), -this->pi / 2));
    EXPECT_TRUE(this->approx_equal(neg_deg.as_deg(), -90));
}

TYPED_TEST(AngleTest, ConversionRadToDeg) {
    // 0 rad = 0 def
    ema::Angle<TypeParam> zero;
    EXPECT_TRUE(this->approx_equal(zero.as_deg(), 0));

    // π rad = 180 deg
    ema::Angle<TypeParam> pi_angle(this->pi);
    EXPECT_TRUE(this->approx_equal(pi_angle.as_deg(), 180));

    // π/2 rad = 90 deg
    ema::Angle<TypeParam> half_pi(this->pi / 2);
    EXPECT_TRUE(this->approx_equal(half_pi.as_deg(), 90));

    // 2π rad = 360 deg
    ema::Angle<TypeParam> two_pi(this->pi * 2);
    EXPECT_TRUE(this->approx_equal(two_pi.as_deg(), 360));
}

TYPED_TEST(AngleTest, ConversionDegToRad) {
    TypeParam degrees = 57.29577951308232; // ~ 1 rad
    auto angle = ema::Angle<TypeParam>::deg(degrees);
    EXPECT_TRUE(this->approx_equal(angle.as_rad(), 1));

    // 180 deg = π rad
    auto angle180 = ema::Angle<TypeParam>::deg(180);
    EXPECT_TRUE(this->approx_equal(angle180.as_rad(), this->pi));

    // 360 deg = 2π rad
    auto angle360 = ema::Angle<TypeParam>::deg(360);
    EXPECT_TRUE(this->approx_equal(angle360.as_rad(), this->pi * 2));
}

TYPED_TEST(AngleTest, Addition) {
    auto a = ema::Angle<TypeParam>::deg(30);
    auto b = ema::Angle<TypeParam>::deg(45);
    auto sum = a + b;

    EXPECT_TRUE(this->approx_equal(sum.as_deg(), 75));
    EXPECT_TRUE(this->approx_equal(sum.as_rad(), ema::Angle<TypeParam>::deg(75).as_rad()));
}

TYPED_TEST(AngleTest, CompoundAddition) {
    auto a = ema::Angle<TypeParam>::deg(30);
    auto b = ema::Angle<TypeParam>::deg(45);

    a += b;
    EXPECT_TRUE(this->approx_equal(a.as_deg(), 75));
}

TYPED_TEST(AngleTest, Subtraction) {
    auto a = ema::Angle<TypeParam>::deg(90);
    auto b = ema::Angle<TypeParam>::deg(30);
    auto diff = a - b;

    EXPECT_TRUE(this->approx_equal(diff.as_deg(), 60));
}

TYPED_TEST(AngleTest, CompoundSubtraction) {
    auto a = ema::Angle<TypeParam>::deg(90);
    auto b = ema::Angle<TypeParam>::deg(30);

    a -= b;
    EXPECT_TRUE(this->approx_equal(a.as_deg(), 60));
}

TYPED_TEST(AngleTest, OperatorSpaceship) {
    auto a = ema::Angle<TypeParam>::deg(30);
    auto b = ema::Angle<TypeParam>::deg(45);
    auto c = ema::Angle<TypeParam>::deg(45);

    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(b == c);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(b >= a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b <= c);
    EXPECT_TRUE(b >= c);
}

TYPED_TEST(AngleTest, MixedRadDegOperations) {
    auto rad_angle = ema::Angle<TypeParam>::rad(this->pi);
    auto deg_angle = ema::Angle<TypeParam>::deg(90);

    auto sum = rad_angle + deg_angle;
    EXPECT_TRUE(this->approx_equal(sum.as_deg(), 270)); // 180 + 90

    auto diff = rad_angle - deg_angle;
    EXPECT_TRUE(this->approx_equal(diff.as_deg(), 90)); // 180 - 90
}

TEST(AngleLiteralsTest, DegreeLiterals) {
    using namespace ema::literals;

    auto angle = 90.0_deg;
    EXPECT_FLOAT_EQ(angle.as_deg(), 90.0f);
    EXPECT_FLOAT_EQ(angle.as_rad(), std::numbers::pi_v<float> / 2);

    auto angle2 = 180.0_deg;
    EXPECT_FLOAT_EQ(angle2.as_deg(), 180.0f);
    EXPECT_FLOAT_EQ(angle2.as_rad(), std::numbers::pi_v<float>);
}

TEST(AngleLiteralsTest, RadianLiterals) {
    using namespace ema::literals;

    auto angle = 3.14159265_rad;
    EXPECT_FLOAT_EQ(angle.as_rad(), std::numbers::pi_v<float>);
    EXPECT_FLOAT_EQ(angle.as_deg(), 180.0f);

    auto angle2 = 1.57079633_rad; // π/2
    EXPECT_FLOAT_EQ(angle2.as_rad(), std::numbers::pi_v<float> / 2);
    EXPECT_FLOAT_EQ(angle2.as_deg(), 90.0f);
}

TEST(AngleLiteralsTest, LiteralOperations) {
    using namespace ema::literals;

    auto sum = 45.0_deg + 45.0_deg;
    EXPECT_FLOAT_EQ(sum.as_deg(), 90.0f);

    auto diff = 180.0_deg - 90.0_deg;
    EXPECT_FLOAT_EQ(diff.as_deg(), 90.0f);

    auto mixed = 90.0_deg + 3.14159265_rad; // 90° + 180° = 270°
    EXPECT_NEAR(mixed.as_deg(), 270.0f, 1e-6);
}

TYPED_TEST(AngleTest, Constants) {
    using namespace ema::constants;

    EXPECT_TRUE(this->approx_equal(zero_angle<TypeParam>.as_rad(), 0));
    EXPECT_TRUE(this->approx_equal(zero_angle<TypeParam>.as_deg(), 0));

    EXPECT_TRUE(this->approx_equal(right_angle<TypeParam>.as_deg(), 90));
    EXPECT_TRUE(this->approx_equal(right_angle<TypeParam>.as_rad(), this->pi / 2));

    EXPECT_TRUE(this->approx_equal(straight_angle<TypeParam>.as_deg(), 180));
    EXPECT_TRUE(this->approx_equal(straight_angle<TypeParam>.as_rad(), this->pi));

    EXPECT_TRUE(this->approx_equal(full_angle<TypeParam>.as_deg(), 360));
    EXPECT_TRUE(this->approx_equal(full_angle<TypeParam>.as_rad(), this->pi * 2));

    EXPECT_TRUE(this->approx_equal(fov_60<TypeParam>.as_deg(), 60));
    EXPECT_TRUE(this->approx_equal(fov_90<TypeParam>.as_deg(), 90));
    EXPECT_TRUE(this->approx_equal(fov_120<TypeParam>.as_deg(), 120));
}

TYPED_TEST(AngleTest, ConstantsOperations) {
    using namespace ema::constants;

    // 90° + 90° = 180°
    auto sum = right_angle<TypeParam> + right_angle<TypeParam>;
    EXPECT_TRUE(this->approx_equal(sum, straight_angle<TypeParam>));

    // 360° - 180° = 180°
    auto diff = full_angle<TypeParam> - straight_angle<TypeParam>;
    EXPECT_TRUE(this->approx_equal(diff, straight_angle<TypeParam>));

    // 120° > 90°
    EXPECT_TRUE(fov_120<TypeParam> > fov_90<TypeParam>);
    EXPECT_TRUE(fov_60<TypeParam> < fov_90<TypeParam>);
}

TYPED_TEST(AngleTest, LargeAngles) {
    auto large_deg = ema::Angle<TypeParam>::deg(720);
    EXPECT_TRUE(this->approx_equal(large_deg.as_deg(), 720));
    EXPECT_TRUE(this->approx_equal(large_deg.as_rad(), this->pi * 4));

    auto large_rad = ema::Angle<TypeParam>::rad(this->pi * 10);
    EXPECT_TRUE(this->approx_equal(large_rad.as_rad(), this->pi * 10));
    EXPECT_TRUE(this->approx_equal(large_rad.as_deg(), 1800));
}

TYPED_TEST(AngleTest, VerySmallAngles) {
    TypeParam tiny = this->epsilon / 10;
    auto tiny_angle = ema::Angle<TypeParam>::rad(tiny);

    EXPECT_TRUE(this->approx_equal(tiny_angle.as_rad(), tiny));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
