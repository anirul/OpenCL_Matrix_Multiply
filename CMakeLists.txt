cmake_minimum_required(VERSION 2.8)

project(OpenCL_Video)

include("cmake/FindCXXFeatures.cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX11_COMPILER_FLAGS}")

if(APPLE)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD c++11)
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY libc++)
    set(CMAKE_OSX_ARCHITECTURES x86_64)
    set(CMAKE_LIBRARY_PREFIX /opt/local /usr/local)

    include_directories(
		/opt/local/include
        ${PROJECT_SOURCE_DIR}/thirdparty
    )

    find_library(OPENCL_LIBRARY OpenCL)

endif(APPLE)

# boost mess
if(WIN32 OR APPLE)
	set(Boost_USE_STATIC_LIBS ON)
endif(WIN32 OR APPLE)

find_package(Boost 1.54.0 REQUIRED COMPONENTS
    program_options
    filesystem
    system
    date_time
    chrono
)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found")
endif()

add_executable(
    OpenCL_Matrix_Multiply
    ${PROJECT_SOURCE_DIR}/sources/main.cpp
	${PROJECT_SOURCE_DIR}/sources/cl_matrix_multiply.cpp
	${PROJECT_SOURCE_DIR}/sources/cl_matrix_multiply.hpp
)

target_link_libraries(
    OpenCL_Matrix_Multiply
    ${Boost_LIBRARIES}
	${OPENCL_LIBRARY}
)
