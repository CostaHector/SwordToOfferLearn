cmake_minimum_required(VERSION 3.10)
project(SwordToOfferTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GOOGLETEST_VERSION "1.14.0")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/googletest)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/googletest/include)

include_directories(../src/header)

add_executable(SwordToOfferTest test_main.cpp ../src/TreeRelated.cpp)
target_link_libraries(SwordToOfferTest PRIVATE gtest gtest_main)


