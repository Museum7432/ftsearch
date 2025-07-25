
add_executable(test_all
    # QueueTest.cpp
    # flatip_test.cpp
    # loop_bench.cpp
    # raggedtest.cpp
    search_test.cpp
)

target_link_libraries(test_all seqr GTest::gtest_main)

gtest_discover_tests(test_all)
