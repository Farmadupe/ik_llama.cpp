cmake_minimum_required(VERSION 3.16)
project(ubatch_predictor_driver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The predictor is self-contained (it pulls in only <cmath>/<cstdint>), so the driver
# compiles its single translation unit straight from ../src -- no dependency on the rest
# of libllama.
add_executable(ubatch_predictor_driver
    main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/llama-ubatch-predictor.cpp
)
target_include_directories(ubatch_predictor_driver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
