#include <gtest/gtest.h>

#include "ema/vec/vec.hpp"

template <typename T>
class VecTest : public ::testing::Test {
  protected:
    using Vec2 = ema::Vec<T, 2>;
    using Vec3 = ema::Vec<T, 3>;
    using Vec4 = ema::Vec<T, 4>;

    const T zero = static_cast<T>(0);
    const T one = static_cast<T>(1);
    const T two = static_cast<T>(2);
    const T three = static_cast<T>(3);
    const T four = static_cast<T>(4);
};

using TestTypes = ::testing::Types<int, float, double>;
TYPED_TEST_SUITE(VecTest, TestTypes);

TYPED_TEST(VecTest, DefaultConstructor) {
    typename TestFixture::Vec2 v2;
    EXPECT_EQ(v2.x(), this->zero);
    EXPECT_EQ(v2.y(), this->zero);

    typename TestFixture::Vec3 v3;
    EXPECT_EQ(v3.x(), this->zero);
    EXPECT_EQ(v3.y(), this->zero);
    EXPECT_EQ(v3.z(), this->zero);

    typename TestFixture::Vec4 v4;
    EXPECT_EQ(v4.x(), this->zero);
    EXPECT_EQ(v4.y(), this->zero);
    EXPECT_EQ(v4.z(), this->zero);
    EXPECT_EQ(v4.w(), this->zero);
}

TYPED_TEST(VecTest, SingleValueConstructor) {
    const TypeParam val = static_cast<TypeParam>(5);
    typename TestFixture::Vec2 v2{val};
    EXPECT_EQ(v2.x(), val);
    EXPECT_EQ(v2.y(), val);

    typename TestFixture::Vec3 v3{val};
    EXPECT_EQ(v3.x(), val);
    EXPECT_EQ(v3.y(), val);
    EXPECT_EQ(v3.z(), val);
}

TYPED_TEST(VecTest, MultiValueConstructor) {
    typename TestFixture::Vec2 v2{this->one, this->two};
    EXPECT_EQ(v2.x(), this->one);
    EXPECT_EQ(v2.y(), this->two);

    typename TestFixture::Vec3 v3{this->one, this->two, this->three};
    EXPECT_EQ(v3.x(), this->one);
    EXPECT_EQ(v3.y(), this->two);
    EXPECT_EQ(v3.z(), this->three);

    typename TestFixture::Vec4 v4{this->one, this->two, this->three, this->four};
    EXPECT_EQ(v4.x(), this->one);
    EXPECT_EQ(v4.y(), this->two);
    EXPECT_EQ(v4.z(), this->three);
    EXPECT_EQ(v4.w(), this->four);
}

TYPED_TEST(VecTest, UnitVectors) {
    auto u0_2 = TestFixture::Vec2::Unit(0);
    EXPECT_EQ(u0_2.x(), this->one);
    EXPECT_EQ(u0_2.y(), this->zero);

    auto u1_2 = TestFixture::Vec2::Unit(1);
    EXPECT_EQ(u1_2.x(), this->zero);
    EXPECT_EQ(u1_2.y(), this->one);

    auto u0_3 = TestFixture::Vec3::Unit(0);
    EXPECT_EQ(u0_3.x(), this->one);
    EXPECT_EQ(u0_3.y(), this->zero);
    EXPECT_EQ(u0_3.z(), this->zero);

    auto u2_3 = TestFixture::Vec3::Unit(2);
    EXPECT_EQ(u2_3.x(), this->zero);
    EXPECT_EQ(u2_3.y(), this->zero);
    EXPECT_EQ(u2_3.z(), this->one);

    auto u3_3 = TestFixture::Vec3::Unit(3); // 3 % 3 = 0
    EXPECT_EQ(u3_3.x(), this->one);
    EXPECT_EQ(u3_3.y(), this->zero);
    EXPECT_EQ(u3_3.z(), this->zero);
}

TYPED_TEST(VecTest, ZeroVector) {
    auto z2 = TestFixture::Vec2::Zero();
    EXPECT_EQ(z2.x(), this->zero);
    EXPECT_EQ(z2.y(), this->zero);

    auto z3 = TestFixture::Vec3::Zero();
    EXPECT_EQ(z3.x(), this->zero);
    EXPECT_EQ(z3.y(), this->zero);
    EXPECT_EQ(z3.z(), this->zero);
}

TYPED_TEST(VecTest, Dimension) {
    typename TestFixture::Vec2 v2;
    EXPECT_EQ(v2.dimension(), 2);

    typename TestFixture::Vec3 v3;
    EXPECT_EQ(v3.dimension(), 3);

    typename TestFixture::Vec4 v4;
    EXPECT_EQ(v4.dimension(), 4);
}

TYPED_TEST(VecTest, SubscriptOperator) {
    typename TestFixture::Vec3 v{this->one, this->two, this->three};

    EXPECT_EQ(v[0], this->one);
    EXPECT_EQ(v[1], this->two);
    EXPECT_EQ(v[2], this->three);

    v[1] = this->four;
    EXPECT_EQ(v.y(), this->four);
}

TYPED_TEST(VecTest, ComponentAccessors) {
    typename TestFixture::Vec3 v3{this->one, this->two, this->three};
    EXPECT_EQ(v3.x(), this->one);
    EXPECT_EQ(v3.y(), this->two);
    EXPECT_EQ(v3.z(), this->three);
    v3.x() = this->four;
    EXPECT_EQ(v3.x(), this->four);

    typename TestFixture::Vec4 v4{this->one, this->two, this->three, this->four};
    EXPECT_EQ(v4.w(), this->four);
    v4.w() = this->one;
    EXPECT_EQ(v4.w(), this->one);
}

TYPED_TEST(VecTest, Iterators) {
    typename TestFixture::Vec3 v{this->one, this->two, this->three};

    TypeParam sum = this->zero;
    for (const auto& comp : v) {
        sum += comp;
    }
    EXPECT_EQ(sum, this->one + this->two + this->three);

    const typename TestFixture::Vec3 cv = v;
    sum = this->zero;
    for (auto it = cv.cbegin(); it != cv.cend(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, this->one + this->two + this->three);

    sum = this->zero;
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
        sum += *it;
    }
    EXPECT_EQ(sum, this->one + this->two + this->three);
}

TYPED_TEST(VecTest, UnaryMinus) {
    typename TestFixture::Vec3 v{this->one, this->two, this->three};
    typename TestFixture::Vec3 inversed = -v;

    EXPECT_EQ(inversed.x(), -this->one);
    EXPECT_EQ(inversed.y(), -this->two);
    EXPECT_EQ(inversed.z(), -this->three);
}

TYPED_TEST(VecTest, CompoundAddition) {
    typename TestFixture::Vec3 v1{this->one, this->two, this->three};
    typename TestFixture::Vec3 v2{this->two, this->three, this->four};

    v1 += v2;
    EXPECT_EQ(v1.x(), this->one + this->two);
    EXPECT_EQ(v1.y(), this->two + this->three);
    EXPECT_EQ(v1.z(), this->three + this->four);
}

TYPED_TEST(VecTest, CompoundSubtraction) {
    typename TestFixture::Vec3 v1{this->one, this->two, this->three};
    typename TestFixture::Vec3 v2{this->two, this->three, this->four};

    v1 -= v2;
    EXPECT_EQ(v1.x(), this->one - this->two);
    EXPECT_EQ(v1.y(), this->two - this->three);
    EXPECT_EQ(v1.z(), this->three - this->four);
}

TYPED_TEST(VecTest, CompoundScalarMultiplication) {
    typename TestFixture::Vec3 v{this->one, this->two, this->three};
    TypeParam scalar = static_cast<TypeParam>(2);

    v *= scalar;
    EXPECT_EQ(v.x(), this->one * scalar);
    EXPECT_EQ(v.y(), this->two * scalar);
    EXPECT_EQ(v.z(), this->three * scalar);
}

TYPED_TEST(VecTest, CompoundScalarDivision) {
    typename TestFixture::Vec3 v{this->two, this->four, static_cast<TypeParam>(6)};
    TypeParam scalar = static_cast<TypeParam>(2);

    v /= scalar;
    EXPECT_EQ(v.x(), this->two / scalar);
    EXPECT_EQ(v.y(), this->four / scalar);
    EXPECT_EQ(v.z(), static_cast<TypeParam>(6) / scalar);
}

TYPED_TEST(VecTest, BinaryAddition) {
    typename TestFixture::Vec3 v1{this->one, this->two, this->three};
    typename TestFixture::Vec3 v2{this->two, this->three, this->four};

    typename TestFixture::Vec3 result = v1 + v2;
    EXPECT_EQ(result.x(), this->one + this->two);
    EXPECT_EQ(result.y(), this->two + this->three);
    EXPECT_EQ(result.z(), this->three + this->four);

    EXPECT_EQ(v1.x(), this->one);
    EXPECT_EQ(v2.x(), this->two);
}

TYPED_TEST(VecTest, BinarySubtraction) {
    typename TestFixture::Vec3 v1{this->one, this->two, this->three};
    typename TestFixture::Vec3 v2{this->two, this->three, this->four};

    typename TestFixture::Vec3 result = v1 - v2;
    EXPECT_EQ(result.x(), this->one - this->two);
    EXPECT_EQ(result.y(), this->two - this->three);
    EXPECT_EQ(result.z(), this->three - this->four);
}

TYPED_TEST(VecTest, BinaryScalarMultiplication) {
    typename TestFixture::Vec3 v{this->one, this->two, this->three};
    TypeParam scalar = static_cast<TypeParam>(3);

    typename TestFixture::Vec3 result1 = v * scalar;
    EXPECT_EQ(result1.x(), this->one * scalar);
    EXPECT_EQ(result1.y(), this->two * scalar);
    EXPECT_EQ(result1.z(), this->three * scalar);

    typename TestFixture::Vec3 result2 = scalar * v;
    EXPECT_EQ(result2.x(), this->one * scalar);
    EXPECT_EQ(result2.y(), this->two * scalar);
    EXPECT_EQ(result2.z(), this->three * scalar);

    // commutativity:
    EXPECT_EQ(result1.x(), result2.x());
    EXPECT_EQ(result1.y(), result2.y());
    EXPECT_EQ(result1.z(), result2.z());
}

TYPED_TEST(VecTest, BinaryScalarDivision) {
    typename TestFixture::Vec3 v(this->two, this->four, static_cast<TypeParam>(6));
    TypeParam scalar = static_cast<TypeParam>(2);

    typename TestFixture::Vec3 result = v / scalar;
    EXPECT_EQ(result.x(), this->two / scalar);
    EXPECT_EQ(result.y(), this->four / scalar);
    EXPECT_EQ(result.z(), static_cast<TypeParam>(6) / scalar);
}

TYPED_TEST(VecTest, ZeroVectorProperties) {
    using Vec3 = typename TestFixture::Vec3;

    Vec3 v(this->one, this->two, this->three);
    Vec3 zero = Vec3::Zero();

    // v + zero = v
    Vec3 result1 = v + zero;
    EXPECT_EQ(result1.x(), v.x());
    EXPECT_EQ(result1.y(), v.y());
    EXPECT_EQ(result1.z(), v.z());

    // zero + v = v
    Vec3 result2 = zero + v;
    EXPECT_EQ(result2.x(), v.x());
    EXPECT_EQ(result2.y(), v.y());
    EXPECT_EQ(result2.z(), v.z());

    // v - v = zero
    Vec3 result3 = v - v;
    EXPECT_EQ(result3.x(), zero.x());
    EXPECT_EQ(result3.y(), zero.y());
    EXPECT_EQ(result3.z(), zero.z());

    // v + (-v) = zero
    Vec3 result4 = v + (-v);
    EXPECT_EQ(result4.x(), zero.x());
    EXPECT_EQ(result4.y(), zero.y());
    EXPECT_EQ(result4.z(), zero.z());
}

TEST(VecAliasesTest, AliasFunctionality) {
    ema::Vec2f v2(1.0f, 2.0f);
    EXPECT_FLOAT_EQ(v2.x(), 1.0f);
    EXPECT_FLOAT_EQ(v2.y(), 2.0f);

    ema::Vec3d v3(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(v3.x(), 1.0);
    EXPECT_DOUBLE_EQ(v3.y(), 2.0);
    EXPECT_DOUBLE_EQ(v3.z(), 3.0);

    ema::Vec4i v4(1, 2, 3, 4);
    EXPECT_EQ(v4.x(), 1);
    EXPECT_EQ(v4.y(), 2);
    EXPECT_EQ(v4.z(), 3);
    EXPECT_EQ(v4.w(), 4);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
