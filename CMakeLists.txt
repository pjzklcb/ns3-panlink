check_include_file_cxx(stdint.h HAVE_STDINT_H)
if(HAVE_STDINT_H)
    add_definitions(-DHAVE_STDINT_H)
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

# we need CONFIG for debug mode in macOS
find_package(Protobuf CONFIG)
if(NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
endif()
include_directories(${Protobuf_INCLUDE_DIRS})

# Check if C++-based ML frameworks are available
# Must do before build_lib (which adds examples/ subdirectory)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/model/libtensorflow)
    message(STATUS "TensorFlow C library found, examples using libtensorflow are enabled")
    set(NS3AI_LIBTENSORFLOW_EXAMPLES ON)
    set(Libtensorflow_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/model/libtensorflow/include")
    if(NOT EXISTS ${Libtensorflow_INCLUDE_DIR})
        message(FATAL_ERROR "Include directory of TensorFlow C library does not exist")
    endif()
    set(Libtensorflow_LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/model/libtensorflow/lib")
    if(NOT EXISTS ${Libtensorflow_LIBRARY_DIR})
        message(FATAL_ERROR "Library directory of TensorFlow C library does not exist")
    else()
        file(GLOB TensorFlow_LIBRARIES "${Libtensorflow_LIBRARY_DIR}/*.so" "${Libtensorflow_LIBRARY_DIR}/*.dylib")
    endif()
else()
    message(STATUS "TensorFlow C library not found, examples using libtensorflow are disabled")
    set(NS3AI_LIBTENSORFLOW_EXAMPLES OFF)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/model/libtorch)
    message(STATUS "PyTorch C++ library found, examples using libtorch are enabled")
    set(NS3AI_LIBTORCH_EXAMPLES ON)
    set(Libtorch_INCLUDE_DIRS
            "${CMAKE_CURRENT_SOURCE_DIR}/model/libtorch/include;${CMAKE_CURRENT_SOURCE_DIR}/model/libtorch/include/torch/csrc/api/include")
    foreach(dir ${Libtorch_INCLUDE_DIRS})
        if(NOT EXISTS ${dir})
            message(FATAL_ERROR "Include directory ${dir} of PyTorch C++ library does not exist")
        endif()
    endforeach ()
    set(Libtorch_LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/model/libtorch/lib")
    if(NOT EXISTS ${Libtorch_LIBRARY_DIR})
        message(FATAL_ERROR "Library directory of PyTorch C++ library does not exist")
    else()
        file(GLOB Torch_LIBRARIES "${Libtorch_LIBRARY_DIR}/*.so" "${Libtorch_LIBRARY_DIR}/*.dylib")
    endif()
else()
    message(STATUS "PyTorch C++ library not found, examples using libtorch are disabled")
    set(NS3AI_LIBTORCH_EXAMPLES OFF)
endif()

add_subdirectory(contrib/ai)
add_subdirectory(contrib/vr-app)

set(source_files
    env/env-monitor.cc
    env/env-agent.cc
    env/cca-env-agent.cc
    env/env-agent-manager.cc
    helper/panlink-helper.cc
    model/panlink-sta-wifi-mac.cc
    utils/app-installer.cc
    utils/intf-deployer.cc
    utils/panlink-utils.cc
)

set(header_files
    env/env-monitor.h
    env/env-agent.h
    env/cca-env-agent.h
    env/env-agent-manager.h
    env/env-utils.h
    env/obs-act-list.h
    helper/panlink-helper.h
    model/panlink-sta-wifi-mac.h
    utils/app-installer.h
    utils/intf-deployer.h
    utils/panlink-utils.h
)

build_lib(
  LIBNAME panlink
  SOURCE_FILES ${source_files}
  HEADER_FILES ${header_files}
  LIBRARIES_TO_LINK 
    
    ${libai}
    ${libapplications}
    ${libcore}
    ${libinternet}
    ${libvr-app}
    ${libwifi}
)

pybind11_add_module(cpp_py_interface env/cpp-py-interface.cc)
set_target_properties(cpp_py_interface PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env)
target_link_libraries(cpp_py_interface PRIVATE
        ${libpanlink}
)