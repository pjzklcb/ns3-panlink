set(gsl_libraries)
if(${GSL_FOUND})
  set(gsl_libraries
      ${GSL_LIBRARIES}
  )
endif()

add_subdirectory(baseline)
