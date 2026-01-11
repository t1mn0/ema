#include "ema/vec/vec.hpp"
#include <gtest/gtest.h>

using namespace ema;

class Vec3Test : public ::testing::Test {
  protected:
    Vec3f v1{1.0f, 2.0f, 3.0f};
    Vec3f v2{4.0f, 5.0f, 6.0f};
    const float eps = 1e-5f;
};

TEST_F(Vec3Test, BasicArithmetic) {
    auto sum = v1 + v2;
    EXPECT_FLOAT_EQ(sum.x(), 5.0f);
    EXPECT_FLOAT_EQ(sum.y(), 7.0f);
    EXPECT_FLOAT_EQ(sum.z(), 9.0f);

    auto diff = v2 - v1;
    EXPECT_EQ(diff[0], 3.0f);
    EXPECT_EQ(diff[1], 3.0f);
    EXPECT_EQ(diff[2], 3.0f);
}
