#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "ema/angle/const.hpp"
#include "ema/mat/mat.hpp"

template <typename T>
class MatFuncTest : public ::testing::Test {
  protected:
    const T epsilon = std::numeric_limits<T>::epsilon() * 100;

    bool approx_equal(T a, T b, T tol_multiplier = 1) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(a - b) <= epsilon * tol_multiplier;
        } else {
            return a == b;
        }
    }

    template <size_t C, size_t R>
    bool approx_equal(const ema::Mat<T, C, R>& a, const ema::Mat<T, C, R>& b, T tol_multiplier = 1) {
        for (size_t col = 0; col < C; ++col) {
            for (size_t row = 0; row < R; ++row) {
                if (!approx_equal(a(row, col), b(row, col), tol_multiplier)) {
                    return false;
                }
            }
        }
        return true;
    }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MatFuncTest, TestTypes);

TYPED_TEST(MatFuncTest, TransposeSquare) {
    ema::Mat<TypeParam, 3, 3> m;
    // [1 2 3]
    // [4 5 6]
    // [7 8 9]
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m(i, j) = static_cast<TypeParam>(i * 3 + j + 1);
        }
    }

    auto transposed = ema::make::transpose(m);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_TRUE(this->approx_equal(transposed(i, j), m(j, i)));
        }
    }

    auto double_transposed = ema::make::transpose(transposed);
    EXPECT_TRUE(this->approx_equal(double_transposed, m));
}

TYPED_TEST(MatFuncTest, TransposeNonSquare) {
    ema::Mat<TypeParam, 3, 2> m;
    // clang-format off
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
    // clang-format on

    auto transposed = ema::make::transpose(m);

    EXPECT_EQ(transposed.col_num(), 2);
    EXPECT_EQ(transposed.row_num(), 3);

    EXPECT_TRUE(this->approx_equal(transposed(0, 0), 1));
    EXPECT_TRUE(this->approx_equal(transposed(0, 1), 4));
    EXPECT_TRUE(this->approx_equal(transposed(1, 0), 2));
    EXPECT_TRUE(this->approx_equal(transposed(1, 1), 5));
    EXPECT_TRUE(this->approx_equal(transposed(2, 0), 3));
    EXPECT_TRUE(this->approx_equal(transposed(2, 1), 6));
}

TYPED_TEST(MatFuncTest, Trace) {
    auto identity = ema::Mat<TypeParam, 3, 3>::Identity();
    EXPECT_TRUE(this->approx_equal(trace(identity), static_cast<TypeParam>(3)));

    auto diag = ema::Mat<TypeParam, 4, 4>::Diagonal(5);
    EXPECT_TRUE(this->approx_equal(trace(diag), static_cast<TypeParam>(20)));

    ema::Mat<TypeParam, 3, 3> m;
    // clang-format off
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
    m(2, 0) = 7; m(2, 1) = 8; m(2, 2) = 9;
    // clang-format on

    EXPECT_TRUE(this->approx_equal(trace(m), static_cast<TypeParam>(15)));
}

TYPED_TEST(MatFuncTest, Determinant2x2) {
    auto identity = ema::Mat<TypeParam, 2, 2>::Identity();
    EXPECT_TRUE(this->approx_equal(ema::determinant(identity), static_cast<TypeParam>(1)));

    ema::Mat<TypeParam, 2, 2> m;
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 4;
    // det = 1*4 - 2*3 = 4 - 6 = -2
    EXPECT_TRUE(this->approx_equal(ema::determinant(m), static_cast<TypeParam>(-2)));
}

TYPED_TEST(MatFuncTest, Determinant3x3) {
    auto identity = ema::Mat<TypeParam, 3, 3>::Identity();
    EXPECT_TRUE(this->approx_equal(ema::determinant(identity), static_cast<TypeParam>(1)));

    ema::Mat<TypeParam, 3, 3> m;
    // [1 2 3]
    // [4 5 6]
    // [7 8 9]
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m(i, j) = static_cast<TypeParam>(i * 3 + j + 1);
        }
    }
    // det = 1*(5*9 - 6*8) - 2*(4*9 - 6*7) + 3*(4*8 - 5*7) =
    //     = 1*(45-48) - 2*(36-42) + 3*(32-35) =
    //     = 1*(-3) - 2*(-6) + 3*(-3) =
    //     = -3 + 12 - 9 = 0
    EXPECT_TRUE(this->approx_equal(ema::determinant(m), static_cast<TypeParam>(0)));
}

TYPED_TEST(MatFuncTest, Minor) {
    ema::Mat<TypeParam, 3, 3> m;
    // [1 2 3]
    // [4 5 6]
    // [7 8 9]
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m(i, j) = static_cast<TypeParam>(i * 3 + j + 1);
        }
    }

    // M[0,0] => [5 6; 8 9] => det = 5*9 - 6*8 = 45-48 = -3
    TypeParam minor_00 = ema::make::minor(m, 0, 0);
    EXPECT_TRUE(this->approx_equal(minor_00, static_cast<TypeParam>(-3)));

    // M[1,2] => [1 2; 7 8] => det = 1*8 - 2*7 = 8-14 = -6
    TypeParam minor_12 = ema::make::minor(m, 1, 2);
    EXPECT_TRUE(this->approx_equal(minor_12, static_cast<TypeParam>(-6)));
}

TYPED_TEST(MatFuncTest, Cofactor) {
    ema::Mat<TypeParam, 3, 3> m;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m(i, j) = static_cast<TypeParam>(i * 3 + j + 1);
        }
    }

    // C[0,0]: (-1)^(0+0) * M[0,0] = 1 * (-3) = -3
    TypeParam cofactor_00 = ema::make::cofactor(m, 0, 0);
    EXPECT_TRUE(this->approx_equal(cofactor_00, static_cast<TypeParam>(-3)));

    // C[0,1]: (-1)^(0+1) * M[0,1] = -1 * M[0,1] = -1 * det([4, 6; 7, 9]) = -1 * (4*9 - 6*7) = 6
    TypeParam cofactor_01 = ema::make::cofactor(m, 0, 1);
    EXPECT_TRUE(this->approx_equal(cofactor_01, static_cast<TypeParam>(6)));

    // C[1,2]: (-1)^(1+2) * M[1,2] = -1 * (-6) = 6
    TypeParam cofactor_12 = ema::make::cofactor(m, 1, 2);
    EXPECT_TRUE(this->approx_equal(cofactor_12, static_cast<TypeParam>(6)));
}

TYPED_TEST(MatFuncTest, Inverse) {
    auto identity = ema::Mat<TypeParam, 3, 3>::Identity();
    auto inv_identity = ema::make::inverse(identity);
    EXPECT_TRUE(this->approx_equal(inv_identity, identity));

    auto diag = ema::Mat<TypeParam, 3, 3>::Diagonal(2);
    auto inv_diag = ema::make::inverse(diag);
    auto expected_diag = ema::Mat<TypeParam, 3, 3>::Diagonal(static_cast<TypeParam>(0.5));
    EXPECT_TRUE(this->approx_equal(inv_diag, expected_diag, 10));

    // A * A⁻¹ = I
    ema::Mat<TypeParam, 3, 3> m;
    // clang-format off
    m(0, 0) = 2; m(0, 1) = 0; m(0, 2) = 1;
    m(1, 0) = 0; m(1, 1) = 3; m(1, 2) = 2;
    m(2, 0) = 1; m(2, 1) = 1; m(2, 2) = 4;
    // clang-format on

    auto inv_m = ema::make::inverse(m);
    auto product = m * inv_m;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_TRUE(this->approx_equal(product(i, j), static_cast<TypeParam>(1), 10));
            } else {
                EXPECT_TRUE(this->approx_equal(product(i, j), static_cast<TypeParam>(0), 10));
            }
        }
    }

    ema::Mat<TypeParam, 3, 3> singular;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            singular(i, j) = static_cast<TypeParam>(i + 1);
        }
    }
    auto inv_singular = ema::make::inverse(singular);
    auto zero = ema::Mat<TypeParam, 3, 3>::Zero();
    EXPECT_TRUE(this->approx_equal(inv_singular, zero));
}

TYPED_TEST(MatFuncTest, TranslationMatrix) {
    ema::Vec<TypeParam, 3> translation_vec{2, 3, 4};
    auto trans_mat = ema::make::translation(translation_vec);

    EXPECT_TRUE(this->approx_equal(trans_mat(0, 0), static_cast<TypeParam>(1)));
    EXPECT_TRUE(this->approx_equal(trans_mat(1, 1), static_cast<TypeParam>(1)));
    EXPECT_TRUE(this->approx_equal(trans_mat(2, 2), static_cast<TypeParam>(1)));
    EXPECT_TRUE(this->approx_equal(trans_mat(3, 3), static_cast<TypeParam>(1)));

    EXPECT_TRUE(this->approx_equal(trans_mat(0, 3), static_cast<TypeParam>(2)));
    EXPECT_TRUE(this->approx_equal(trans_mat(1, 3), static_cast<TypeParam>(3)));
    EXPECT_TRUE(this->approx_equal(trans_mat(2, 3), static_cast<TypeParam>(4)));

    ema::Vec<TypeParam, 4> point{1, 1, 1, 1};
    ema::Vec<TypeParam, 4> translated = trans_mat * point;

    EXPECT_TRUE(this->approx_equal(translated[0], static_cast<TypeParam>(3))); // 1 + 2
    EXPECT_TRUE(this->approx_equal(translated[1], static_cast<TypeParam>(4))); // 1 + 3
    EXPECT_TRUE(this->approx_equal(translated[2], static_cast<TypeParam>(5))); // 1 + 4
    EXPECT_TRUE(this->approx_equal(translated[3], static_cast<TypeParam>(1))); // w is 1
}

TYPED_TEST(MatFuncTest, LookAtMatrix) {
    ema::Vec<TypeParam, 3> eye{0, 0, 0};
    ema::Vec<TypeParam, 3> target{0, 0, -1};
    ema::Vec<TypeParam, 3> up{0, 1, 0};

    auto view_mat = ema::make::look_at(eye, target, up);

    EXPECT_TRUE(this->approx_equal(view_mat(3, 3), static_cast<TypeParam>(1)));

    EXPECT_TRUE(this->approx_equal(view_mat(0, 3), static_cast<TypeParam>(0)));
    EXPECT_TRUE(this->approx_equal(view_mat(1, 3), static_cast<TypeParam>(0)));
    EXPECT_TRUE(this->approx_equal(view_mat(2, 3), static_cast<TypeParam>(0)));
}

TYPED_TEST(MatFuncTest, PerspectiveMatrix) {
    TypeParam fov = static_cast<TypeParam>(M_PI / 2.0);    // 90 deg
    TypeParam aspect = static_cast<TypeParam>(16.0 / 9.0); // 16:9
    TypeParam near = static_cast<TypeParam>(0.1);
    TypeParam far = static_cast<TypeParam>(100.0);

    auto proj_mat = ema::make::perspective(ema::Angle<TypeParam>(fov), aspect, near, far);

    TypeParam tan_half_fov = std::tan(fov / 2);
    TypeParam expected_00 = static_cast<TypeParam>(1) / (aspect * tan_half_fov);
    TypeParam expected_11 = static_cast<TypeParam>(1) / tan_half_fov;

    EXPECT_TRUE(this->approx_equal(proj_mat(0, 0), expected_00, 10));
    EXPECT_TRUE(this->approx_equal(proj_mat(1, 1), expected_11, 10));
    EXPECT_TRUE(this->approx_equal(proj_mat(2, 2), -(far + near) / (far - near), 10));
    EXPECT_TRUE(this->approx_equal(proj_mat(2, 3), -(2 * far * near) / (far - near), 10));
    EXPECT_TRUE(this->approx_equal(proj_mat(3, 2), static_cast<TypeParam>(-1)));
}

TYPED_TEST(MatFuncTest, OrthographicMatrix) {
    TypeParam left = -10, right = 10;
    TypeParam bottom = -5, top = 5;
    TypeParam near = 0.1, far = 100;

    auto ortho_mat = ema::make::ortho(left, right, bottom, top, near, far);

    EXPECT_TRUE(this->approx_equal(ortho_mat(0, 0), static_cast<TypeParam>(2) / (right - left)));
    EXPECT_TRUE(this->approx_equal(ortho_mat(1, 1), static_cast<TypeParam>(2) / (top - bottom)));
    EXPECT_TRUE(this->approx_equal(ortho_mat(2, 2), -static_cast<TypeParam>(2) / (far - near)));

    EXPECT_TRUE(this->approx_equal(ortho_mat(0, 3), -(right + left) / (right - left)));
    EXPECT_TRUE(this->approx_equal(ortho_mat(1, 3), -(top + bottom) / (top - bottom)));
    EXPECT_TRUE(this->approx_equal(ortho_mat(2, 3), -(far + near) / (far - near)));

    EXPECT_TRUE(ortho_mat(0, 0) > 0);
    EXPECT_TRUE(ortho_mat(1, 1) > 0);
    EXPECT_TRUE(ortho_mat(2, 2) < 0);
}

TYPED_TEST(MatFuncTest, IsSymmetric) {
    ema::Mat<TypeParam, 3, 3> symmetric;
    symmetric(0, 0) = 1;
    symmetric(0, 1) = 2;
    symmetric(0, 2) = 3;
    symmetric(1, 0) = 2;
    symmetric(1, 1) = 4;
    symmetric(1, 2) = 5;
    symmetric(2, 0) = 3;
    symmetric(2, 1) = 5;
    symmetric(2, 2) = 6;

    EXPECT_TRUE(ema::is_symmetric(symmetric));

    ema::Mat<TypeParam, 3, 3> non_symmetric;
    non_symmetric(0, 0) = 1;
    non_symmetric(0, 1) = 2;
    non_symmetric(0, 2) = 3;
    non_symmetric(1, 0) = 4;
    non_symmetric(1, 1) = 5;
    non_symmetric(1, 2) = 6; // 4 ≠ 2
    non_symmetric(2, 0) = 7;
    non_symmetric(2, 1) = 8;
    non_symmetric(2, 2) = 9;

    EXPECT_FALSE(ema::is_symmetric(non_symmetric));

    auto identity = ema::Mat<TypeParam, 3, 3>::Identity();
    EXPECT_TRUE(ema::is_symmetric(identity));

    auto diag = ema::Mat<TypeParam, 3, 3>::Diagonal(5);
    EXPECT_TRUE(ema::is_symmetric(diag));
}

TYPED_TEST(MatFuncTest, MatrixProperties) {
    // det(Aᵀ) = det(A)
    ema::Mat<TypeParam, 3, 3> m;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m(i, j) = static_cast<TypeParam>(i * 3 + j + 1);
        }
    }

    auto transposed = ema::make::transpose(m);
    TypeParam det_m = ema::determinant(m);
    TypeParam det_transposed = ema::determinant(transposed);

    EXPECT_TRUE(this->approx_equal(det_m, det_transposed));

    // det(A⁻¹) = 1/det(A)
    if (std::abs(det_m) > this->epsilon * 100) {
        auto inv_m = ema::make::inverse(m);
        TypeParam det_inv = ema::determinant(inv_m);

        EXPECT_TRUE(this->approx_equal(det_inv * det_m, static_cast<TypeParam>(1), 100));
    }

    // trace(A + B) = trace(A) + trace(B)
    ema::Mat<TypeParam, 3, 3> a, b;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            a(i, j) = static_cast<TypeParam>(i * 3 + j + 1);
            b(i, j) = static_cast<TypeParam>((i * 3 + j + 1) * 2);
        }
    }

    TypeParam trace_sum = trace(a + b);
    TypeParam sum_traces = trace(a) + trace(b);

    EXPECT_TRUE(this->approx_equal(trace_sum, sum_traces));
}

TYPED_TEST(MatFuncTest, RotationMatrix2DWithAngle) {
    auto rot_90 = ema::make::rotation_matrix_2d(ema::Angle<TypeParam>::deg(90.0));

    // 90°: [0, -1; 1, 0]
    EXPECT_TRUE(this->approx_equal(rot_90(0, 0), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_90(0, 1), TypeParam(-1), 10));
    EXPECT_TRUE(this->approx_equal(rot_90(1, 0), TypeParam(1), 10));
    EXPECT_TRUE(this->approx_equal(rot_90(1, 1), TypeParam(0), 10));

    auto angle45 = ema::Angle<TypeParam>::deg(45.0);
    auto rot_45 = ema::make::rotation_matrix_2d(angle45);

    TypeParam c45 = std::cos(angle45.as_rad());
    TypeParam s45 = std::sin(angle45.as_rad());

    EXPECT_TRUE(this->approx_equal(rot_45(0, 0), c45, 10));
    EXPECT_TRUE(this->approx_equal(rot_45(0, 1), -s45, 10));
    EXPECT_TRUE(this->approx_equal(rot_45(1, 0), s45, 10));
    EXPECT_TRUE(this->approx_equal(rot_45(1, 1), c45, 10));

    ema::Vec<TypeParam, 2> v{1, 0};
    ema::Vec<TypeParam, 2> rotated = rot_90 * v;

    EXPECT_TRUE(this->approx_equal(rotated[0], TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rotated[1], TypeParam(1), 10));

    auto rot_90_twice = rot_90 * rot_90;
    auto rot_180 = ema::make::rotation_matrix_2d(ema::Angle<TypeParam>::deg(180.0));
    EXPECT_TRUE(this->approx_equal(rot_90_twice, rot_180, 10));
}

TYPED_TEST(MatFuncTest, RotationMatrixXWithAngle) {
    auto rot_x = ema::make::rotation_matrix_x(ema::Angle<TypeParam>::deg(90.0));

    EXPECT_TRUE(this->approx_equal(rot_x(0, 0), TypeParam(1), 10));
    EXPECT_TRUE(this->approx_equal(rot_x(0, 1), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_x(0, 2), TypeParam(0), 10));

    EXPECT_TRUE(this->approx_equal(rot_x(1, 0), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_x(1, 1), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_x(1, 2), TypeParam(-1), 10));

    EXPECT_TRUE(this->approx_equal(rot_x(2, 0), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_x(2, 1), TypeParam(1), 10));
    EXPECT_TRUE(this->approx_equal(rot_x(2, 2), TypeParam(0), 10));

    ema::Vec<TypeParam, 3> v{0, 1, 0};
    ema::Vec<TypeParam, 3> rotated = rot_x * v;

    EXPECT_TRUE(this->approx_equal(rotated[0], TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rotated[1], TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rotated[2], TypeParam(1), 10));

    auto rot_x_inv = ema::make::rotation_matrix_x(ema::Angle<TypeParam>::deg(-90.0));
    auto should_be_identity = rot_x * rot_x_inv;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_TRUE(this->approx_equal(should_be_identity(i, j), TypeParam(1), 10));
            } else {
                EXPECT_TRUE(this->approx_equal(should_be_identity(i, j), TypeParam(0), 10));
            }
        }
    }
}

TYPED_TEST(MatFuncTest, RotationMatrixYWithAngle) {
    auto rot_y = ema::make::rotation_matrix_y(ema::Angle<TypeParam>::deg(90.0));

    EXPECT_TRUE(this->approx_equal(rot_y(0, 0), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_y(0, 1), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_y(0, 2), TypeParam(1), 10));

    EXPECT_TRUE(this->approx_equal(rot_y(1, 0), TypeParam(0)));
    EXPECT_TRUE(this->approx_equal(rot_y(1, 1), TypeParam(1), 10));
    EXPECT_TRUE(this->approx_equal(rot_y(1, 2), TypeParam(0), 10));

    EXPECT_TRUE(this->approx_equal(rot_y(2, 0), TypeParam(-1), 10));
    EXPECT_TRUE(this->approx_equal(rot_y(2, 1), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_y(2, 2), TypeParam(0), 10));

    ema::Vec<TypeParam, 3> v{0, 0, 1};
    ema::Vec<TypeParam, 3> rotated = rot_y * v;

    EXPECT_TRUE(this->approx_equal(rotated[0], TypeParam(1), 10));
    EXPECT_TRUE(this->approx_equal(rotated[1], TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rotated[2], TypeParam(0), 10));
}

TYPED_TEST(MatFuncTest, RotationMatrixZWithAngle) {
    auto rot_z = ema::make::rotation_matrix_z(ema::Angle<TypeParam>::deg(90.0));

    EXPECT_TRUE(this->approx_equal(rot_z(0, 0), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_z(0, 1), TypeParam(-1), 10));
    EXPECT_TRUE(this->approx_equal(rot_z(0, 2), TypeParam(0), 10));

    EXPECT_TRUE(this->approx_equal(rot_z(1, 0), TypeParam(1), 10));
    EXPECT_TRUE(this->approx_equal(rot_z(1, 1), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_z(1, 2), TypeParam(0), 10));

    EXPECT_TRUE(this->approx_equal(rot_z(2, 0), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_z(2, 1), TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rot_z(2, 2), TypeParam(1), 10));

    ema::Vec<TypeParam, 3> v{1, 0, 0};
    ema::Vec<TypeParam, 3> rotated = rot_z * v;

    EXPECT_TRUE(this->approx_equal(rotated[0], TypeParam(0), 10));
    EXPECT_TRUE(this->approx_equal(rotated[1], TypeParam(1), 10));
    EXPECT_TRUE(this->approx_equal(rotated[2], TypeParam(0), 10));
}

TYPED_TEST(MatFuncTest, RotationMatrixAroundAxis) {
    ema::Vec<TypeParam, 3> axis_x{1, 0, 0};
    auto rot_axis = ema::make::rotation_matrix(ema::Angle<TypeParam>::deg(180.0), axis_x);
    auto rot_x_180 = ema::make::rotation_matrix_x(ema::Angle<TypeParam>::deg(180.0));

    EXPECT_TRUE(this->approx_equal(rot_axis, rot_x_180, 10));

    ema::Vec<TypeParam, 3> axis_y{0, 1, 0};
    auto rot_axis_y = ema::make::rotation_matrix(ema::Angle<TypeParam>::deg(90.0), axis_y);
    auto rot_y_90 = ema::make::rotation_matrix_y(ema::Angle<TypeParam>::deg(90.0));

    EXPECT_TRUE(this->approx_equal(rot_axis_y, rot_y_90, 10));

    ema::Vec<TypeParam, 3> axis{1, 1, 0};
    auto angle = ema::Angle<TypeParam>::deg(45.0);
    auto rot_arbitrary = ema::make::rotation_matrix(angle, axis);

    // R * Rᵀ = I
    auto transposed = ema::make::transpose(rot_arbitrary);
    auto should_be_identity = rot_arbitrary * transposed;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_TRUE(this->approx_equal(should_be_identity(i, j), TypeParam(1), 10));
            } else {
                EXPECT_TRUE(this->approx_equal(should_be_identity(i, j), TypeParam(0), 20));
            }
        }
    }

    ema::Vec<TypeParam, 3> norm_axis = ema::normalize(axis);
    ema::Vec<TypeParam, 3> rotated_axis = rot_arbitrary * norm_axis;

    EXPECT_TRUE(this->approx_equal(rotated_axis[0], norm_axis[0], 10));
    EXPECT_TRUE(this->approx_equal(rotated_axis[1], norm_axis[1], 10));
    EXPECT_TRUE(this->approx_equal(rotated_axis[2], norm_axis[2], 10));
}

TYPED_TEST(MatFuncTest, PerspectiveWithAngleConstants) {
    using namespace ema::constants;

    TypeParam aspect = static_cast<TypeParam>(4.0 / 3.0);
    TypeParam near = static_cast<TypeParam>(0.5);
    TypeParam far = static_cast<TypeParam>(1000.0);

    auto proj90 = ema::make::perspective(ema::constants::fov_90<TypeParam>, aspect, near, far);
    auto proj60 = ema::make::perspective(fov_60<TypeParam>, aspect, near, far);

    EXPECT_TRUE(proj90(0, 0) < proj60(0, 0));
    EXPECT_TRUE(proj90(1, 1) < proj60(1, 1));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
