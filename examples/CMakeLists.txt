
build_lib_example(
        NAME ns3channel-access
        SOURCE_FILES
            channel-access.cc
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

# pybind11_add_module(ns3ai_multibss_py multi_bss_py.cc)
# set_target_properties(ns3ai_multibss_py PROPERTIES
#         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(ns3ai_multibss_py PRIVATE
#         ${libai}
# )

# Build Python interface along with C++ lib
# add_dependencies(ns3ai_multibss ns3ai_multibss_py)
