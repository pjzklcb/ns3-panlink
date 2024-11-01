check_include_file_cxx(stdint.h HAVE_STDINT_H)
if(HAVE_STDINT_H)
    add_definitions(-DHAVE_STDINT_H)
endif()

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

set(gsl_libraries)
if(${GSL_FOUND})
  set(gsl_libraries
      ${GSL_LIBRARIES}
  )
endif()

add_subdirectory(contrib/ai)
add_subdirectory(contrib/vr-app)

set(source_files
    env/env-monitor.cc
    env/env-app-tracer.cc
    env/env-mac-tracer.cc
    env/env-phy-tracer.cc
    env/env-agent.cc
    env/cca-env-agent.cc
    env/env-agent-manager.cc
    env/env-utils.cc
    helper/env-helper.cc
    helper/env-agent-helper.cc
    helper/panlink-mac-helper.cc
    model/panlink-ap-wifi-mac.cc
    model/panlink-frame-exchange-manager.cc
    model/panlink-sta-wifi-mac.cc
    model/sensing-mac.cc
    utils/app-installer.cc
    utils/intf-deployer.cc
    utils/panlink-utils.cc
)

set(header_files
    env/env-monitor.h
    env/env-app-tracer.h
    env/env-mac-tracer.h
    env/env-phy-tracer.h
    env/env-agent.h
    env/cca-env-agent.h
    env/env-agent-manager.h
    env/env-utils.h
    env/obs-act-list.h
    helper/env-helper.h
    helper/env-agent-helper.h
    helper/panlink-mac-helper.h
    model/panlink-ap-wifi-mac.h
    model/panlink-frame-exchange-manager.h
    model/panlink-sta-wifi-mac.h
    model/sensing-mac.h
    utils/app-installer.h
    utils/intf-deployer.h
    utils/panlink-utils.h
)

build_lib(
  LIBNAME panlink
  SOURCE_FILES ${source_files}
  HEADER_FILES ${header_files}
  LIBRARIES_TO_LINK 
    ${libapplications}
    ${libcore}
    ${libinternet}
    ${libvr-app}
    ${libwifi}
    ${gsl_libraries}
)

pybind11_add_module(cpp_py_interface env/cpp-py-interface.cc)
set_target_properties(cpp_py_interface PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env)
target_link_libraries(cpp_py_interface PRIVATE
        ${libpanlink}
)