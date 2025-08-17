cmake_minimum_required(VERSION 3.10)
project(SwordToOfferTest)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
# 显式声明Debug模式的编译标志
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GOOGLETEST_VERSION "1.14.0")

set(SRV_PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)

file(GLOB TEST_SRC_CPP # GLOB or GLOB_RECURSE
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB SRV_SRC_CPP
    ${SRV_PROJECT_ROOT_PATH}/src/*.cpp
)

set(TEST_HEADER_PATH
    ${SRV_PROJECT_ROOT_PATH}/src
)

file(GLOB TEST_HEADER_FILES
    ${SRV_PROJECT_ROOT_PATH}/src/*.h
)


# debug message below
message("\n> CMAKE_CURRENT_SOURCE_DIR[${CMAKE_CURRENT_SOURCE_DIR}] ")
message("\n> CMAKE_SOURCE_DIR[${CMAKE_SOURCE_DIR}]")

list(LENGTH TEST_SRC_CPP test_cpp_count)
message("\n> ${test_cpp_count} file(s) in TEST_SRC_CPP, there are:")
foreach(file ${TEST_SRC_CPP})
    message("${file}")
endforeach()

list(LENGTH SRV_SRC_CPP src_cpp_count)
message("\n> ${src_cpp_count} file(s) in SRV_SRC_CPP, there are:")
foreach(file ${SRV_SRC_CPP})
    message("${file}")
endforeach()

list(LENGTH TEST_HEADER_PATH header_count)
message("\n> ${header_count} folder(s) in TEST_HEADER_PATH, there are:")
foreach(h_folder ${TEST_HEADER_PATH})
    message("${h_folder}")
endforeach()
# debug message above

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/googletest)
add_executable(SwordToOfferTest
    ${SRV_SRC_CPP}
    ${TEST_SRC_CPP})
target_sources(SwordToOfferTest PUBLIC ${TEST_HEADER_FILES})

target_include_directories(SwordToOfferTest PRIVATE ${TEST_HEADER_PATH})
target_link_libraries(SwordToOfferTest PRIVATE gtest gtest_main)


