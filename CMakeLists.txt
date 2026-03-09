cmake_minimum_required(VERSION 3.10)

# 1. Project name and language
project(aruconano_perf LANGUAGES CXX)

# 2. Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 3. Handle Build Type Portability (Release mode)
# ---------------------------------------------------------------------------

# For Linux/Unix (Single-config): Default to Release if nothing is specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# 4. Find Dependencies
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# 5. Add Executable
add_executable(testperf testperf.cpp aruco_nano.h json.hpp)

# 6. Link Libraries
target_link_libraries(testperf PRIVATE ${OpenCV_LIBS})

 



# ---------------------------------------------------------------------------
# 8. Status Printout
# ---------------------------------------------------------------------------
message(STATUS "---------------------------------------------------")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler ID:    ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "---------------------------------------------------")
