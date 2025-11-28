cmake_minimum_required(VERSION 3.14)
project(dmc_route 
    VERSION 0.1.0 
    DESCRIPTION "Differentiable Muskingum-Cunge Routing"
    LANGUAGES CXX
)

# ============ Options ============
option(DMC_ENABLE_AD "Enable automatic differentiation via CoDiPack" ON)
option(DMC_ENABLE_NETCDF "Enable NetCDF input support" ON)
option(DMC_BUILD_TESTS "Build unit tests" ON)
option(DMC_BUILD_PYTHON "Build Python bindings via pybind11" OFF)
option(DMC_BUILD_SHARED "Build shared library" ON)

# ============ C++ Standard ============
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============ Dependencies via FetchContent ============
include(FetchContent)

# Use shallow clones to speed up downloads
set(FETCHCONTENT_QUIET OFF)

# CoDiPack (header-only AD library)
if(DMC_ENABLE_AD)
    FetchContent_Declare(
        codipack
        GIT_REPOSITORY https://github.com/SciCompKL/CoDiPack.git
        GIT_TAG v2.2.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(codipack)
    message(STATUS "CoDiPack enabled for automatic differentiation")
endif()

# OpenMP for parallel routing
option(DMC_ENABLE_OPENMP "Enable OpenMP parallelization" OFF)
if(DMC_ENABLE_OPENMP)
    # On macOS with AppleClang, we need to help CMake find libomp from Homebrew
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(HOMEBREW_LIBOMP_PREFIX)
            message(STATUS "Found Homebrew libomp at: ${HOMEBREW_LIBOMP_PREFIX}")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        else()
            message(FATAL_ERROR "OpenMP requested but libomp not found. Install with: brew install libomp")
        endif()
    endif()
    
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP enabled for parallel routing")
    
    # CRITICAL: Enable thread-safe CoDiPack when using OpenMP
    # CoDiPack v2.2+ uses thread-local tapes by default, but we add
    # explicit defines for safety and compatibility
    add_compile_definitions(CODI_EnableOpenMP)
endif()

# BMI C++ specification (header-only, just download it)
set(BMI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/bmi")
if(NOT EXISTS "${BMI_INCLUDE_DIR}/bmi.hxx")
    message(STATUS "Downloading BMI C++ header...")
    file(MAKE_DIRECTORY ${BMI_INCLUDE_DIR})
    file(DOWNLOAD
        "https://raw.githubusercontent.com/csdms/bmi-cxx/master/bmi.hxx"
        "${BMI_INCLUDE_DIR}/bmi.hxx"
        STATUS download_status
    )
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download bmi.hxx")
    endif()
endif()

# yaml-cpp for configuration - prefer system install
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found, will download (this may take a minute)...")
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.7.0
        GIT_SHALLOW TRUE
    )
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yaml-cpp)
endif()

# NetCDF for reading model outputs
if(DMC_ENABLE_NETCDF)
    # Try to find system netCDF-cxx4
    find_package(netCDFCxx QUIET)
    if(NOT netCDFCxx_FOUND)
        # Fall back to netCDF C library with our own C++ wrapper
        find_package(netCDF QUIET)
        if(netCDF_FOUND)
            message(STATUS "Found netCDF C library, using thin C++ wrapper")
            set(DMC_USE_NETCDF_C ON)
        else()
            message(WARNING "NetCDF not found. Install libnetcdf-dev and libnetcdf-c++4-dev, or disable with -DDMC_ENABLE_NETCDF=OFF")
            set(DMC_ENABLE_NETCDF OFF)
        endif()
    else()
        message(STATUS "Found netCDF-cxx4")
        set(DMC_USE_NETCDF_CXX4 ON)
    endif()
endif()

# nlohmann/json for GeoJSON parsing
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# ============ Main Library ============
add_library(dmc_route INTERFACE)

target_include_directories(dmc_route INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${BMI_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

if(DMC_ENABLE_AD)
    target_include_directories(dmc_route INTERFACE
        ${codipack_SOURCE_DIR}/include
    )
    target_compile_definitions(dmc_route INTERFACE DMC_USE_CODIPACK)
endif()

# NetCDF support
if(DMC_ENABLE_NETCDF)
    target_compile_definitions(dmc_route INTERFACE DMC_USE_NETCDF)
    if(DMC_USE_NETCDF_CXX4)
        target_link_libraries(dmc_route INTERFACE netCDF::netcdf-cxx4)
        target_compile_definitions(dmc_route INTERFACE DMC_USE_NETCDF_CXX4)
    elseif(DMC_USE_NETCDF_C)
        target_link_libraries(dmc_route INTERFACE netCDF::netcdf)
        target_compile_definitions(dmc_route INTERFACE DMC_USE_NETCDF_C)
    endif()
endif()

# JSON for GeoJSON parsing
target_link_libraries(dmc_route INTERFACE nlohmann_json::nlohmann_json)

# ============ Shared Library Wrapper ============
if(DMC_BUILD_SHARED)
    add_library(dmc_route_shared SHARED
        src/lib.cpp
    )
    target_link_libraries(dmc_route_shared PUBLIC dmc_route)
    set_target_properties(dmc_route_shared PROPERTIES
        OUTPUT_NAME dmc_route
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# ============ Python Bindings (pybind11) ============
option(DMC_BUILD_PYTHON "Build Python bindings" OFF)
if(DMC_BUILD_PYTHON)
    # Find pybind11
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    message(STATUS "Building Python bindings")
    
    # Create the Python module
    pybind11_add_module(pydmc_route python/bindings.cpp)
    target_link_libraries(pydmc_route PRIVATE dmc_route)
    
    # Set output name and properties
    set_target_properties(pydmc_route PROPERTIES
        OUTPUT_NAME pydmc_route
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    )
    
    # Install Python module
    if(DEFINED SKBUILD)
        # If building with scikit-build, install to the package directory
        install(TARGETS pydmc_route DESTINATION .)
    else()
        # Otherwise install to lib/pythonX.Y/site-packages
        install(TARGETS pydmc_route 
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
        )
    endif()
    
    message(STATUS "Python module will be built as: pydmc_route")
endif()

# ============ Executable for Testing ============
add_executable(dmc_route_run src/main.cpp)
target_link_libraries(dmc_route_run PRIVATE dmc_route)

# ============ Tests ============
if(DMC_BUILD_TESTS)
    enable_testing()
    
    # Gradient verification test
    add_executable(test_gradients tests/test_gradients.cpp)
    target_link_libraries(test_gradients PRIVATE dmc_route)
    add_test(NAME gradient_verification COMMAND test_gradients)
    
    # Single reach routing test
    add_executable(test_single_reach tests/test_single_reach.cpp)
    target_link_libraries(test_single_reach PRIVATE dmc_route)
    add_test(NAME single_reach_routing COMMAND test_single_reach)
    
    # BMI interface test
    add_executable(test_bmi tests/test_bmi.cpp)
    target_link_libraries(test_bmi PRIVATE dmc_route)
    add_test(NAME bmi_interface COMMAND test_bmi)
    
    # Rigorous gradient verification suite (AD vs Finite Differences)
    add_executable(test_gradient_verification tests/test_gradient_verification.cpp)
    target_link_libraries(test_gradient_verification PRIVATE dmc_route)
    add_test(NAME gradient_verification_suite COMMAND test_gradient_verification)
    
    # Comprehensive test suite
    add_executable(test_comprehensive tests/test_comprehensive.cpp)
    target_link_libraries(test_comprehensive PRIVATE dmc_route)
    add_test(NAME comprehensive_suite COMMAND test_comprehensive)
endif()

# OpenMP linking for parallel routing
if(DMC_ENABLE_OPENMP)
    target_link_libraries(dmc_route INTERFACE OpenMP::OpenMP_CXX)
    target_link_libraries(dmc_route_run PRIVATE OpenMP::OpenMP_CXX)
    if(DMC_BUILD_TESTS)
        target_link_libraries(test_comprehensive PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# ============ Python Bindings ============
if(DMC_BUILD_PYTHON)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(pydmc_route python/bindings.cpp)
        target_link_libraries(pydmc_route PRIVATE dmc_route)
    else()
        message(WARNING "pybind11 not found, skipping Python bindings")
    endif()
endif()

# ============ Installation ============
include(GNUInstallDirs)

install(DIRECTORY include/dmc
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(DMC_BUILD_SHARED)
    install(TARGETS dmc_route_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

install(TARGETS dmc_route_run
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# ============ Package Configuration ============
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dmc_routeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dmc_routeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dmc_routeConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dmc_route
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dmc_routeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dmc_routeConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dmc_route
)
