cmake_minimum_required(VERSION 3.20)

project(sih26147_native VERSION 0.2.0 LANGUAGES CXX)

option(SIH_BUILD_PYTHON "Build the pybind11 extension" ON)
option(SIH_BUILD_TESTS "Build native tests" OFF)
option(SIH_PROBE_FUTURE_DEPENDENCIES "Report availability of planned DSP/FEC dependencies" ON)
option(SIH_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(SIH_ENABLE_SANITIZERS)
    if(MSVC)
        message(FATAL_ERROR "SIH_ENABLE_SANITIZERS currently requires a Clang/GCC-compatible compiler")
    endif()
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

if(SIH_PROBE_FUTURE_DEPENDENCIES)
    include(cmake/DependencyProbes.cmake)
    sih_probe_future_dependencies()
endif()

add_library(sih_core STATIC
    cpp/src/core/types.cpp
    cpp/src/dsp/preprocessing.cpp
    cpp/src/dsp/spectral.cpp
    cpp/src/estimation/analysis.cpp
    cpp/src/receiver/receiver.cpp
    cpp/src/bitstream/bitstream.cpp
    cpp/src/fec/viterbi.cpp
    cpp/src/io/recording_source.cpp
)
add_library(sih::core ALIAS sih_core)

target_compile_features(sih_core PUBLIC cxx_std_20)
target_include_directories(sih_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(sih_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(MSVC)
    target_compile_options(sih_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(sih_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SIH_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_native MODULE
        cpp/bindings/module.cpp
        cpp/bindings/phase2_bindings.cpp
        cpp/bindings/phase3_bindings.cpp
        cpp/bindings/phase4_bindings.cpp
        cpp/bindings/phase5_bindings.cpp
    )
    target_link_libraries(_native PRIVATE sih::core)
    target_compile_features(_native PRIVATE cxx_std_20)
    target_compile_definitions(_native PRIVATE SIH_NATIVE_VERSION="${PROJECT_VERSION}.dev1")
    install(TARGETS _native
        LIBRARY DESTINATION signal_analysis
        RUNTIME DESTINATION signal_analysis
    )
endif()

include(CTest)
if(SIH_BUILD_TESTS)
    enable_testing()
    add_executable(sih_core_tests cpp/tests/test_viterbi.cpp)
    target_link_libraries(sih_core_tests PRIVATE sih::core)
    target_compile_features(sih_core_tests PRIVATE cxx_std_20)
    add_test(NAME sih_core_viterbi COMMAND sih_core_tests)

    add_executable(sih_phase2_tests cpp/tests/test_phase2.cpp)
    target_link_libraries(sih_phase2_tests PRIVATE sih::core)
    target_compile_features(sih_phase2_tests PRIVATE cxx_std_20)
    add_test(NAME sih_phase2_core COMMAND sih_phase2_tests)

    add_executable(sih_phase3_tests cpp/tests/test_phase3.cpp)
    target_link_libraries(sih_phase3_tests PRIVATE sih::core)
    target_compile_features(sih_phase3_tests PRIVATE cxx_std_20)
    add_test(NAME sih_phase3_estimation COMMAND sih_phase3_tests)

    add_executable(sih_phase4_tests cpp/tests/test_phase4.cpp)
    target_link_libraries(sih_phase4_tests PRIVATE sih::core)
    target_compile_features(sih_phase4_tests PRIVATE cxx_std_20)
    add_test(NAME sih_phase4_receiver COMMAND sih_phase4_tests)

    add_executable(sih_phase5_tests cpp/tests/test_phase5.cpp)
    target_link_libraries(sih_phase5_tests PRIVATE sih::core)
    target_compile_features(sih_phase5_tests PRIVATE cxx_std_20)
    add_test(NAME sih_phase5_bitstream COMMAND sih_phase5_tests)
endif()
