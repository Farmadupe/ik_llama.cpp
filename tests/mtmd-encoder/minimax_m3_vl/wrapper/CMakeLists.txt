find_package(Threads REQUIRED)

# Distinct target name so this parity wrapper can coexist with other vision
# parity wrappers in the tree (each builds its own ik_encode-style driver).
add_executable(ik_encode_minimax_m3_vl ik_encode.cpp)
target_link_libraries     (ik_encode_minimax_m3_vl PRIVATE common mtmd Threads::Threads)
target_include_directories(ik_encode_minimax_m3_vl PRIVATE ${CMAKE_SOURCE_DIR}/vendor/stb)  # stb_image.h
target_compile_features   (ik_encode_minimax_m3_vl PRIVATE cxx_std_17)
