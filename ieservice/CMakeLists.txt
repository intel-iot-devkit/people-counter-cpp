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

cmake_minimum_required (VERSION 2.8)

project(ieservice)

if(DEFINED ENV{DLSDKROOT})
    message("Using DL SDK include path $ENV{DLSDKROOT}")
elseif(DEFINED ENV{INTEL_CVSDK_DIR})
    message("Using CV SDK include path $ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine")
    set(ENV{DLSDKROOT} $ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine)
else()
    message(FATAL_ERROR " *** Neither DLSDKROOT or INTEL_CVSDK_DIR are defined *** ")
endif()

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
####################################

if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set (ARCH intel64)
else()
    set (ARCH ia32)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

#linux OS name detection
if (UNIX AND NOT(LIB_FOLDER))
    SET(LIB_DL dl)

    if (NOT EXISTS "/etc/lsb-release")
        execute_process(COMMAND find /etc/ -maxdepth 1 -type f -name *-release -exec cat {} \;
                    OUTPUT_VARIABLE release_data RESULT_VARIABLE result)
        set(name_regex "NAME=\"([^ \"\n]*).*\"\n")
        set(version_regex "VERSION=\"([0-9]+(\\.[0-9]+)?)[^\n]*\"")
    else()
        #linux version detection using cat /etc/lsb-release
        file(READ "/etc/lsb-release" release_data)
        set(name_regex "DISTRIB_ID=([^ \n]*)\n")
        set(version_regex "DISTRIB_RELEASE=([0-9]+(\\.[0-9]+)?)")
    endif()

    string(REGEX MATCH ${name_regex} name ${release_data})
    set(os_name ${CMAKE_MATCH_1})

    string(REGEX MATCH ${version_regex} version ${release_data})
    set(os_name "${os_name} ${CMAKE_MATCH_1}")

    if (NOT os_name)
        message(FATAL_ERROR "Cannot detect OS via reading /etc/*-release:\n ${release_data}")
    endif()

    message (STATUS "/etc/*-release distrib: ${os_name}")

    if (${os_name} STREQUAL "Ubuntu 14.04")
        set(OS_LIB_FOLDER "ubuntu_14.04/")
    elseif (${os_name} STREQUAL "Ubuntu 16.04")
        set(OS_LIB_FOLDER "ubuntu_16.04/")
    elseif (${os_name} STREQUAL "CentOS 7")
        set(OS_LIB_FOLDER "centos_7.2/")
    else()
        message(FATAL_ERROR "${os_name} is not supported. List of supported OS: Ubuntu 14.04, Ubuntu 16.04, CentOS 7")
    endif()

    add_definitions(-DOS_LIB_FOLDER="${OS_LIB_FOLDER}")
endif()

if (NOT(BIN_FOLDER))
    set (BIN_FOLDER bin/${ARCH})
endif()

if (NOT(LIB_FOLDER))
    set (LIB_FOLDER ../lib/${OS_LIB_FOLDER}${ARCH})
endif()

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -ffast-math ${CMAKE_CXX_FLAGS}")

if(NOT(UNIX))
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else ()
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)
endif()

if (WIN32)
    if(NOT "${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()

    set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
    if (ENABLE_OMP)
        find_package(OpenMP)
        if (OPENMP_FOUND)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
	if (APPLE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
	elseif(UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self -Wmaybe-uninitialized")
	endif()
endif()

set(InferenceEngine_INCLUDE_DIRS $ENV{DLSDKROOT}/include)
list(APPEND CMAKE_MODULE_PATH "$ENV{DLSDKROOT}/samples/cmake")
include(feature_defs)

add_subdirectory($ENV{DLSDKROOT}/samples/extension ${CMAKE_BINARY_DIR}/extension)
add_subdirectory(thirdparty/gflags)
add_subdirectory(sample)

