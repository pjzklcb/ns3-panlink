check_include_file_cxx(stdint.h HAVE_STDINT_H)
if(HAVE_STDINT_H)
    add_definitions(-DHAVE_STDINT_H)
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

add_subdirectory(contrib/ai)
add_subdirectory(contrib/vr-app)

set(source_files
    env/env-monitor.cc
    model/panlink-sta-wifi-mac.cc
    utils/app-installer.cc
    utils/intf-deployer.cc
    utils/panlink-utils.cc
)

set(header_files
    env/obs-act-type.h
    env/packet-info.h
    env/env-monitor.h
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