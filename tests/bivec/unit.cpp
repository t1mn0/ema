#include <cmath>

#include <gtest/gtest.h>

#include "ema/bivec/bivec3.hpp"

template <typename T>
class BiVecTest : public ::testing::Test {
  protected:
    const T epsilon = std::numeric_limits<T>::epsilon() * 100;

    bool approx_equal(T a, T b) { return std::abs(a - b) <= epsilon; }

    bool approx_equal(const ema::BiVec3<T>& a, const ema::BiVec3<T>& b) {
        return approx_equal(a.xy(), b.xy()) &&
               approx_equal(a.yz(), b.yz()) &&
               approx_equal(a.zx(), b.zx());
    }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(BiVecTest, TestTypes);

TYPED_TEST(BiVecTest, DefaultConstructor) {
    ema::BiVec3<TypeParam> b;
    EXPECT_TRUE(this->approx_equal(b.xy(), 0));
    EXPECT_TRUE(this->approx_equal(b.yz(), 0));
    EXPECT_TRUE(this->approx_equal(b.zx(), 0));
}

TYPED_TEST(BiVecTest, ComponentConstructor) {
    ema::BiVec3<TypeParam> b{1, 2, 3};
    EXPECT_TRUE(this->approx_equal(b.xy(), 1));
    EXPECT_TRUE(this->approx_equal(b.yz(), 2));
    EXPECT_TRUE(this->approx_equal(b.zx(), 3));
}

TYPED_TEST(BiVecTest, UnaryMinus) {
    ema::BiVec3<TypeParam> b{1, 2, 3};
    ema::BiVec3<TypeParam> neg = -b;

    EXPECT_TRUE(this->approx_equal(neg.xy(), -1));
    EXPECT_TRUE(this->approx_equal(neg.yz(), -2));
    EXPECT_TRUE(this->approx_equal(neg.zx(), -3));
}

TYPED_TEST(BiVecTest, ScalarMultiplication) {
    ema::BiVec3<TypeParam> b{1, 2, 3};

    b *= 2;
    EXPECT_TRUE(this->approx_equal(b.xy(), 2));
    EXPECT_TRUE(this->approx_equal(b.yz(), 4));
    EXPECT_TRUE(this->approx_equal(b.zx(), 6));

    ema::BiVec3<TypeParam> b2 = b * static_cast<TypeParam>(0.5);
    EXPECT_TRUE(this->approx_equal(b2.xy(), 1));
    EXPECT_TRUE(this->approx_equal(b2.yz(), 2));
    EXPECT_TRUE(this->approx_equal(b2.zx(), 3));

    ema::BiVec3<TypeParam> b3 = static_cast<TypeParam>(0.5) * b;
    EXPECT_TRUE(this->approx_equal(b3.xy(), 1));
    EXPECT_TRUE(this->approx_equal(b3.yz(), 2));
    EXPECT_TRUE(this->approx_equal(b3.zx(), 3));
}

TYPED_TEST(BiVecTest, Addition) {
    ema::BiVec3<TypeParam> a{1, 2, 3};
    ema::BiVec3<TypeParam> b{4, 5, 6};

    ema::BiVec3<TypeParam> sum = a + b;
    EXPECT_TRUE(this->approx_equal(sum.xy(), 5));
    EXPECT_TRUE(this->approx_equal(sum.yz(), 7));
    EXPECT_TRUE(this->approx_equal(sum.zx(), 9));
}

TYPED_TEST(BiVecTest, Subtraction) {
    ema::BiVec3<TypeParam> a{4, 5, 6};
    ema::BiVec3<TypeParam> b{1, 2, 3};

    ema::BiVec3<TypeParam> diff = a - b;
    EXPECT_TRUE(this->approx_equal(diff.xy(), 3));
    EXPECT_TRUE(this->approx_equal(diff.yz(), 3));
    EXPECT_TRUE(this->approx_equal(diff.zx(), 3));
}

TYPED_TEST(BiVecTest, WedgeProductBasic) {
    ema::Vec<TypeParam, 3> i{1, 0, 0};
    ema::Vec<TypeParam, 3> j{0, 1, 0};
    ema::Vec<TypeParam, 3> k{0, 0, 1};

    // i ∧ j = e₁∧e₂ basis
    ema::BiVec3<TypeParam> ij = wedge(i, j);
    EXPECT_TRUE(this->approx_equal(ij.xy(), 1));
    EXPECT_TRUE(this->approx_equal(ij.yz(), 0));
    EXPECT_TRUE(this->approx_equal(ij.zx(), 0));

    // j ∧ k = e₂∧e₃ basis
    ema::BiVec3<TypeParam> jk = wedge(j, k);
    EXPECT_TRUE(this->approx_equal(jk.xy(), 0));
    EXPECT_TRUE(this->approx_equal(jk.yz(), 1));
    EXPECT_TRUE(this->approx_equal(jk.zx(), 0));

    // k ∧ i = e₃∧e₁ basis
    ema::BiVec3<TypeParam> ki = wedge(k, i);
    EXPECT_TRUE(this->approx_equal(ki.xy(), 0));
    EXPECT_TRUE(this->approx_equal(ki.yz(), 0));
    EXPECT_TRUE(this->approx_equal(ki.zx(), 1));

    // i ∧ i = e₁∧e₁ = 0
    ema::BiVec3<TypeParam> ii = wedge(i, i);
    EXPECT_TRUE(this->approx_equal(ii.xy(), 0));
    EXPECT_TRUE(this->approx_equal(ii.yz(), 0));
    EXPECT_TRUE(this->approx_equal(ii.zx(), 0));
}

TYPED_TEST(BiVecTest, WedgeProductAnticommutativity) {
    ema::Vec<TypeParam, 3> a{1, 2, 3};
    ema::Vec<TypeParam, 3> b{4, 5, 6};

    // anticommutativity: a ∧ b = -(b ∧ a)
    ema::BiVec3<TypeParam> ab = wedge(a, b);
    ema::BiVec3<TypeParam> ba = wedge(b, a);

    EXPECT_TRUE(this->approx_equal(ab.xy(), -ba.xy()));
    EXPECT_TRUE(this->approx_equal(ab.yz(), -ba.yz()));
    EXPECT_TRUE(this->approx_equal(ab.zx(), -ba.zx()));
}

TYPED_TEST(BiVecTest, WedgeProductBilinearity) {
    ema::Vec<TypeParam, 3> a{1, 2, 3};
    ema::Vec<TypeParam, 3> b{4, 5, 6};
    ema::Vec<TypeParam, 3> c{7, 8, 9};
    TypeParam scalar = static_cast<TypeParam>(2.5);

    // linearity in first argument:
    ema::BiVec3<TypeParam> left1 = wedge(a + b, c);
    ema::BiVec3<TypeParam> left2 = wedge(a, c) + wedge(b, c);
    EXPECT_TRUE(this->approx_equal(left1, left2));

    // linearity in second argument:
    ema::BiVec3<TypeParam> right1 = wedge(a, b + c);
    ema::BiVec3<TypeParam> right2 = wedge(a, b) + wedge(a, c);
    EXPECT_TRUE(this->approx_equal(right1, right2));

    // homogeneity
    ema::BiVec3<TypeParam> scalar1 = wedge(scalar * a, b);
    ema::BiVec3<TypeParam> scalar2 = scalar * wedge(a, b);
    ema::BiVec3<TypeParam> scalar3 = wedge(a, scalar * b);
    EXPECT_TRUE(this->approx_equal(scalar1, scalar2));
    EXPECT_TRUE(this->approx_equal(scalar2, scalar3));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
