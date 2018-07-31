# Copyright (c) 2018 Intel Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 2.8)

set (TARGET_NAME "obj_recognition")

set(OpenCV_STATIC OFF)
find_package(OpenCV 3.2 REQUIRED core imgproc imgcodecs highgui videoio)

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        )

if (UNIX)
    SET(LIB_DL dl)
endif()

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

# Properties->C/C++->General->Additional Include Directories
include_directories (
        ${OpenCV_INCLUDE_DIRS}
        $ENV{DLSDKROOT}/include
        $ENV{DLSDKROOT}/samples/extension
        ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/gflags/include)

link_directories(
        #${CMAKE_SOURCE_DIR}/${LIB_FOLDER}
        #$ENV{DLSDKROOT}/lib/${ARCH}/${CMAKE_BUILD_TYPE}
        $ENV{DLSDKROOT}/lib/ubuntu_16.04/${ARCH})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS} mqtt.cpp)

add_dependencies(${TARGET_NAME} gflags cpu_extension)

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE")

target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS} inference_engine cpu_extension gflags paho-mqtt3cs)

if(UNIX)
    target_link_libraries( ${TARGET_NAME} ${LIB_DL} pthread)
elseif(WIN32)
    target_link_libraries( ${TARGET_NAME} ${LIB_DL} shlwapi)
endif()
