#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "ema/mat/mat.hpp"

template <typename T>
class MatTest : public ::testing::Test {
  protected:
    const T epsilon = std::numeric_limits<T>::epsilon() * 100;

    bool approx_equal(T a, T b) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(a - b) <= epsilon;
        } else {
            return a == b;
        }
    }

    template <size_t C, size_t R>
    bool approx_equal(const ema::Mat<T, C, R>& a, const ema::Mat<T, C, R>& b) {
        for (size_t col = 0; col < C; ++col) {
            for (size_t row = 0; row < R; ++row) {
                if (!approx_equal(a(row, col), b(row, col))) {
                    return false;
                }
            }
        }
        return true;
    }
};

using TestTypes = ::testing::Types<int, float, double>;
TYPED_TEST_SUITE(MatTest, TestTypes);

TYPED_TEST(MatTest, DefaultConstructor) {
    ema::Mat<TypeParam, 2, 2> m;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_TRUE(this->approx_equal(m(i, j), static_cast<TypeParam>(0)));
        }
    }
}

TYPED_TEST(MatTest, ScalarConstructor) {
    ema::Mat<TypeParam, 3, 3> m{5};

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_TRUE(this->approx_equal(m(i, j), static_cast<TypeParam>(5)));
        }
    }
}

TYPED_TEST(MatTest, IdentityMatrix) {
    auto identity = ema::Mat<TypeParam, 3, 3>::Identity();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_TRUE(this->approx_equal(identity(i, j), static_cast<TypeParam>(1)));
            } else {
                EXPECT_TRUE(this->approx_equal(identity(i, j), static_cast<TypeParam>(0)));
            }
        }
    }
}

TYPED_TEST(MatTest, ZeroMatrix) {
    auto zero = ema::Mat<TypeParam, 3, 3>::Zero();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_TRUE(this->approx_equal(zero(i, j), static_cast<TypeParam>(0)));
        }
    }
}

TYPED_TEST(MatTest, DiagonalMatrix) {
    auto diag = ema::Mat<TypeParam, 4, 4>::Diagonal(7);

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_TRUE(this->approx_equal(diag(i, j), static_cast<TypeParam>(7)));
            } else {
                EXPECT_TRUE(this->approx_equal(diag(i, j), static_cast<TypeParam>(0)));
            }
        }
    }
}

TYPED_TEST(MatTest, IsSquare) {
    ema::Mat<TypeParam, 3, 3> square;
    ema::Mat<TypeParam, 2, 3> non_square;

    EXPECT_TRUE(square.is_square());
    EXPECT_FALSE(non_square.is_square());
}

TYPED_TEST(MatTest, ElementAccess) {
    ema::Mat<TypeParam, 3, 3> m;

    m(0, 0) = 1;
    m(1, 1) = 2;
    m(2, 2) = 3;
    m(0, 2) = 4;

    EXPECT_TRUE(this->approx_equal(m(0, 0), static_cast<TypeParam>(1)));
    EXPECT_TRUE(this->approx_equal(m(1, 1), static_cast<TypeParam>(2)));
    EXPECT_TRUE(this->approx_equal(m(2, 2), static_cast<TypeParam>(3)));
    EXPECT_TRUE(this->approx_equal(m(0, 2), static_cast<TypeParam>(4)));

    const auto& cm = m;
    EXPECT_TRUE(this->approx_equal(cm(0, 0), static_cast<TypeParam>(1)));
}

TYPED_TEST(MatTest, ColumnAccess) {
    ema::Mat<TypeParam, 3, 3> m;

    m.col(0) = ema::Vec<TypeParam, 3>{1, 2, 3};
    m.col(1) = ema::Vec<TypeParam, 3>{4, 5, 6};
    m.col(2) = ema::Vec<TypeParam, 3>{7, 8, 9};

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_TRUE(this->approx_equal(m(j, i), static_cast<TypeParam>(i * 3 + j + 1)));
        }
    }
}

TYPED_TEST(MatTest, Iterators) {
    ema::Mat<TypeParam, 3, 3> m;
    m.col(0) = ema::Vec<TypeParam, 3>{1, 2, 3};
    m.col(1) = ema::Vec<TypeParam, 3>{4, 5, 6};
    m.col(2) = ema::Vec<TypeParam, 3>{7, 8, 9};

    TypeParam sum = 0;
    for (const auto& col : m) {
        for (const auto& elem : col) {
            sum += elem;
        }
    }
    EXPECT_TRUE(this->approx_equal(sum, static_cast<TypeParam>(45)));

    const auto& cm = m;
    sum = 0;
    for (auto it = cm.begin(); it != cm.end(); ++it) {
        for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
            sum += *it2;
        }
    }
    EXPECT_TRUE(this->approx_equal(sum, static_cast<TypeParam>(45)));
}

TYPED_TEST(MatTest, CompoundAddition) {
    ema::Mat<TypeParam, 2, 2> a{1};
    ema::Mat<TypeParam, 2, 2> b{2};

    a += b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_TRUE(this->approx_equal(a(i, j), static_cast<TypeParam>(3)));
        }
    }
}

TYPED_TEST(MatTest, CompoundSubtraction) {
    ema::Mat<TypeParam, 2, 2> a{5};
    ema::Mat<TypeParam, 2, 2> b{2};

    a -= b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_TRUE(this->approx_equal(a(i, j), static_cast<TypeParam>(3)));
        }
    }
}

TYPED_TEST(MatTest, CompoundScalarMultiplication) {
    ema::Mat<TypeParam, 3, 3> a;
    // clang-format off
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
    a(2, 0) = 7; a(2, 1) = 8; a(2, 2) = 9;
    // clang-format on

    a *= 2;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_TRUE(this->approx_equal(a(i, j), 2 * static_cast<TypeParam>(i * 3 + j + 1)));
        }
    }
}

TYPED_TEST(MatTest, MatrixVectorMultiplication) {
    ema::Mat<TypeParam, 3, 3> m;
    // [1 2 3]
    // [4 5 6]
    // [7 8 9]
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m(i, j) = static_cast<TypeParam>(i * 3 + j + 1);
        }
    }

    ema::Vec<TypeParam, 3> v{1, 2, 3};
    ema::Vec<TypeParam, 3> result = m * v;

    // expected:
    // [1*1 + 2*2 + 3*3] = [14]
    // [4*1 + 5*2 + 6*3] = [32]
    // [7*1 + 8*2 + 9*3] = [50]
    EXPECT_TRUE(this->approx_equal(result[0], static_cast<TypeParam>(14)));
    EXPECT_TRUE(this->approx_equal(result[1], static_cast<TypeParam>(32)));
    EXPECT_TRUE(this->approx_equal(result[2], static_cast<TypeParam>(50)));

    auto identity = ema::Mat<TypeParam, 3, 3>::Identity();
    ema::Vec<TypeParam, 3> v2{2, 3, 4};
    ema::Vec<TypeParam, 3> result2 = identity * v2;

    EXPECT_TRUE(this->approx_equal(result2[0], v2[0]));
    EXPECT_TRUE(this->approx_equal(result2[1], v2[1]));
    EXPECT_TRUE(this->approx_equal(result2[2], v2[2]));

    auto zero = ema::Mat<TypeParam, 3, 3>::Zero();
    ema::Vec<TypeParam, 3> result3 = zero * v2;

    EXPECT_TRUE(this->approx_equal(result3[0], static_cast<TypeParam>(0)));
    EXPECT_TRUE(this->approx_equal(result3[1], static_cast<TypeParam>(0)));
    EXPECT_TRUE(this->approx_equal(result3[2], static_cast<TypeParam>(0)));
}

TYPED_TEST(MatTest, NonSquareMatrixVectorMultiplication) {
    ema::Mat<TypeParam, 3, 2> m;
    // [1 2 3]
    // [4 5 6]
    // clang-format off
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
    // clang-format on

    ema::Vec<TypeParam, 3> v{1, 2, 3};
    ema::Vec<TypeParam, 2> result = m * v;

    // expected:
    // [1*1 + 2*2 + 3*3] = [14]
    // [4*1 + 5*2 + 6*3] = [32]
    EXPECT_TRUE(this->approx_equal(result[0], static_cast<TypeParam>(14)));
    EXPECT_TRUE(this->approx_equal(result[1], static_cast<TypeParam>(32)));
}

TYPED_TEST(MatTest, MatrixMultiplicationSquare) {
    ema::Mat<TypeParam, 3, 3> a;
    ema::Mat<TypeParam, 3, 3> b;

    // [1 2 3]
    // [4 5 6]
    // [7 8 9]
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            a(i, j) = static_cast<TypeParam>(i * 3 + j + 1);
        }
    }

    // B = 2 * A
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            b(i, j) = static_cast<TypeParam>((i * 3 + j + 1) * 2);
        }
    }

    auto result1 = a * b;
    // B = 2 * A => A * B = A * (2A) = 2(A * A)
    // [1 2 3]   [1 2 3]   [30  36  42]
    // [4 5 6] X [4 5 6] = [66  81  96]
    // [7 8 9]   [7 8 9]   [102 126 150]
    //
    // => A^2 * 2
    // [60  72  84]
    // [132 162 192]
    // [204 252 300]
    ema::Mat<TypeParam, 3, 3> expected;
    // clang-format off
    expected(0, 0) = 60;  expected(0, 1) = 72;  expected(0, 2) = 84;
    expected(1, 0) = 132; expected(1, 1) = 162; expected(1, 2) = 192;
    expected(2, 0) = 204; expected(2, 1) = 252; expected(2, 2) = 300;
    // clang-format on

    EXPECT_TRUE(this->approx_equal(result1, expected));

    auto identity = ema::Mat<TypeParam, 3, 3>::Identity();
    auto result2 = a * identity;

    // A * I = A
    EXPECT_TRUE(this->approx_equal(result2, a));

    // I * A = A
    auto result3 = identity * a;
    EXPECT_TRUE(this->approx_equal(result3, a));

    // A * 0 = 0
    auto zero = ema::Mat<TypeParam, 3, 3>::Zero();
    auto result4 = a * zero;
    EXPECT_TRUE(this->approx_equal(result4, zero));
}

TYPED_TEST(MatTest, MatrixMultiplicationNonSquare) {
    // M(2x3) * M(3x2) = M(2x2)
    ema::Mat<TypeParam, 3, 2> a; // 2 rows, 3 cols
    ema::Mat<TypeParam, 2, 3> b; // 3 rows, 2 cols

    // A (2x3):
    // [1 2 3]
    // [4 5 6]
    // clang-format off
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
    // clang-format on

    // B (3x2):
    // [ 7  8]
    // [ 9 10]
    // [11 12]
    // clang-format off
    b(0, 0) = 7;  b(0, 1) = 8;
    b(1, 0) = 9;  b(1, 1) = 10;
    b(2, 0) = 11; b(2, 1) = 12;
    // clang-format on

    auto result = a * b;

    EXPECT_EQ(result.col_num(), 2);
    EXPECT_EQ(result.row_num(), 2);

    // result(0,0): 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // result(0,1): 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // result(1,0): 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // result(1,1): 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

    EXPECT_TRUE(this->approx_equal(result(0, 0), static_cast<TypeParam>(58)));
    EXPECT_TRUE(this->approx_equal(result(0, 1), static_cast<TypeParam>(64)));
    EXPECT_TRUE(this->approx_equal(result(1, 0), static_cast<TypeParam>(139)));
    EXPECT_TRUE(this->approx_equal(result(1, 1), static_cast<TypeParam>(154)));
}

TYPED_TEST(MatTest, MatrixMultiplicationProperties) {
    // associativity: (AB)C = A(BC)
    ema::Mat<TypeParam, 3, 3> a{2};
    ema::Mat<TypeParam, 3, 3> b{3};
    ema::Mat<TypeParam, 3, 3> c{4};

    auto left = (a * b) * c;
    auto right = a * (b * c);

    EXPECT_TRUE(this->approx_equal(left, right));

    // distributivity: A(B + C) = AB + AC
    auto left2 = a * (b + c);
    auto right2 = a * b + a * c;

    EXPECT_TRUE(this->approx_equal(left2, right2));

    // (A + B)C = AC + BC
    auto left3 = (a + b) * c;
    auto right3 = a * c + b * c;

    EXPECT_TRUE(this->approx_equal(left3, right3));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
