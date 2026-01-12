include(GoogleTest)

add_executable(vec_class_tests vec/unit/class.cpp)
add_executable(vec_func_tests vec/unit/func.cpp)

target_link_libraries(vec_class_tests PRIVATE emathlib GTest::gtest_main)
target_link_libraries(vec_func_tests PRIVATE emathlib GTest::gtest_main)

gtest_discover_tests(vec_class_tests)
gtest_discover_tests(vec_func_tests)
