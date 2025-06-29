cmake_minimum_required(VERSION 3.10)
project(SwordToOffer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB ROOT_SRC_CPP
    *.cpp
)

file(GLOB SRV_SRC_CPP
    src/*.cpp
)

set(SRV_HEADER_PATH
    src/header
)

list(LENGTH ROOT_SRC_CPP root_cpp_count)
message("\n> ${root_cpp_count} file(s) in ROOT_SRC_CPP, there are:")
foreach(file ${ROOT_SRC_CPP})
    message("${file}")
endforeach()

list(LENGTH SRV_SRC_CPP src_cpp_count)
message("\n> ${src_cpp_count} file(s) in SRV_SRC_CPP, there are:")
foreach(file ${SRV_SRC_CPP})
    message("${file}")
endforeach()

list(LENGTH SRV_HEADER_PATH header_count)
message("\n> ${header_count} folder(s) in SRV_HEADER_PATH, there are:")
foreach(file ${SRV_HEADER_PATH})
    message("${file}")
endforeach()

add_subdirectory(testcase)
add_executable(SwordToOffer ${ROOT_SRC_CPP} ${SRV_SRC_CPP})
target_include_directories(SwordToOffer PUBLIC ${SRV_HEADER_PATH})


