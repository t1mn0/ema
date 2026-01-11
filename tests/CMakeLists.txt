add_executable(vec_tests vec/unit.cpp)

target_link_libraries(vec_tests PRIVATE emathlib GTest::gtest_main)
