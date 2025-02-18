cmake_minimum_required(VERSION 3.15)
project(gloom VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Enable testing
include(CTest)
enable_testing()

# Find required packages
find_package(Eigen3 3.3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

# Set include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Collect source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/core/*.cpp"
    "src/utils/*.cpp"
)

# Create library target
add_library(gloom STATIC ${SOURCES})
target_link_libraries(gloom 
    PUBLIC 
        Eigen3::Eigen
        spdlog::spdlog
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

# Create executable target
add_executable(gloom_exe src/main.cpp)
target_link_libraries(gloom_exe PRIVATE gloom)
set_target_properties(gloom_exe PROPERTIES OUTPUT_NAME gloom)

# Examples
add_subdirectory(examples)

# Tests
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS gloom
    EXPORT gloom-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/gloom
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Export targets
install(EXPORT gloom-targets
    FILE gloom-targets.cmake
    NAMESPACE gloom::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gloom
)

# Create and install config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gloom-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gloom-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gloom
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gloom-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gloom-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gloom-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gloom
)

# Examples CMakeLists.txt
add_subdirectory(examples/basic)
add_subdirectory(examples/advanced)

# Add custom targets
add_custom_target(format
    COMMAND clang-format -i ${SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Formatting source files..."
)

add_custom_target(docs
    COMMAND doxygen Doxyfile
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating documentation..."
)

# Options
option(GLOOM_BUILD_TESTS "Build tests" ON)
option(GLOOM_BUILD_EXAMPLES "Build examples" ON)
option(GLOOM_ENABLE_ASAN "Enable Address Sanitizer" OFF)
option(GLOOM_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" OFF)

# Sanitizer configuration
if(GLOOM_ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

if(GLOOM_ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

# Version information
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gloom/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/gloom/version.hpp
)

# Package configuration
set(CPACK_PACKAGE_NAME "gloom")
set(CPACK_PACKAGE_VENDOR "Gloom")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Gloom - AI Framework")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
include(CPack)