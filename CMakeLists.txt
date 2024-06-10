check_include_file_cxx(stdint.h HAVE_STDINT_H)
if(HAVE_STDINT_H)
    add_definitions(-DHAVE_STDINT_H)
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

set(source_files
    model/panlink-sta-wifi-mac.cc
)

set(header_files
    env/panlink-obs-act.h
    model/panlink-sta-wifi-mac.h
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

pybind11_add_module(panlink_py_interface env/panlink_py_interface.cc)
set_target_properties(panlink_py_interface PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env)
target_link_libraries(panlink_py_interface PRIVATE
        ${libpanlink}
)