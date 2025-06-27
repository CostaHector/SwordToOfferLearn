cmake_minimum_required(VERSION 3.10)
project(SwordToOffer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GOOGLETEST_VERSION "1.14.0")

include_directories(src/header)


add_executable(SwordToOffer main.cpp src/TreeRelated.cpp)

target_sources(SwordToOffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/header/Others.h)


