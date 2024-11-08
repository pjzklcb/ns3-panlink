set(gsl_libraries)
if(${GSL_FOUND})
  set(gsl_libraries
      ${GSL_LIBRARIES}
  )
endif()

add_subdirectory(cca)
add_subdirectory(channel-access)
