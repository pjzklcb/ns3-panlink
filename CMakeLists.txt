set(source_files
    model/panlink-sta-wifi-mac.cc
)

set(header_files
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
