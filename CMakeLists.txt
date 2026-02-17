cmake_minimum_required(VERSION 3.10)

# 1. Name your project
project(aruconano_perf)

# 2. Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. Find the OpenCV package
# This looks for OpenCVConfig.cmake on your system
find_package(OpenCV REQUIRED)

# 4. Add the executable
add_executable(testperf testperf.cpp)

# 5. Link the OpenCV libraries to your executable
target_link_libraries(testperf ${OpenCV_LIBS})

# Optional: Print OpenCV version to the console during configuration
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
