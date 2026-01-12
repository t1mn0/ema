include(GoogleTest)

add_executable(angle_unit_tests angle/unit.cpp)
add_executable(vec_unit_class_tests vec/unit/class.cpp)
add_executable(vec_unit_func_tests vec/unit/func.cpp)

target_link_libraries(angle_unit_tests PRIVATE emathlib GTest::gtest_main)
target_link_libraries(vec_unit_class_tests PRIVATE emathlib GTest::gtest_main)
target_link_libraries(vec_unit_func_tests PRIVATE emathlib GTest::gtest_main)

gtest_discover_tests(angle_unit_tests)
gtest_discover_tests(vec_unit_class_tests)
gtest_discover_tests(vec_unit_func_tests)
