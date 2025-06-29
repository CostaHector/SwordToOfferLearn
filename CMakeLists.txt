cmake_minimum_required(VERSION 3.10)
project(SwordToOffer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SRV_SRC_CPP
    main.cpp
    src/TreeRelated.cpp)
set(SRV_HEADER_PATH
    src/header
)

add_subdirectory(testcase)
add_executable(SwordToOffer ${SRV_SRC_CPP})
target_include_directories(SwordToOffer PUBLIC ${SRV_HEADER_PATH})


