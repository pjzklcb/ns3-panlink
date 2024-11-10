set(gsl_libraries)
if(${GSL_FOUND})
  set(gsl_libraries
      ${GSL_LIBRARIES}
  )
endif()

build_lib_example(
        NAME panlink_cca
        SOURCE_FILES
            cca.cc
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
