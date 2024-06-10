
build_lib_example(
        NAME ns3channel_access
        SOURCE_FILES
            multi-agent-ul.cc
            tgax-residential-propagation-loss-model.cc
        LIBRARIES_TO_LINK
            ${libai}
            ${libcore}
            ${libwifi}
            ${libinternet}
            ${libnetwork}
            ${libvr-app}
            ${libapplications}
            ${libpropagation}
            ${libenergy}
            ${libspectrum}
            ${libantenna}
            ${libbuildings}
            ${libmobility}
            ${libpanlink}
            ${gsl_libraries}
)

pybind11_add_module(panlink_py_interface panlink_py_interface.cc)
set_target_properties(panlink_py_interface PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(panlink_py_interface PRIVATE
        ${libpanlink}
)

# Build Python interface along with C++ lib
add_dependencies(ns3channel_access panlink_py_interface)
