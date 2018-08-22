include(RegexUtils)
test_escape_string_as_regex()

file(GLOB Eigen_directory_files "*")

escape_string_as_regex(ESCAPED_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

foreach(f ${Eigen_directory_files})
  if(NOT f MATCHES "\\.txt" AND NOT f MATCHES "${ESCAPED_CMAKE_CURRENT_SOURCE_DIR}/[.].+" AND NOT f MATCHES "${ESCAPED_CMAKE_CURRENT_SOURCE_DIR}/src")
    list(APPEND Eigen_directory_files_to_install ${f})
  endif()
endforeach(f ${Eigen_directory_files})

# install(FILES
#   ${Eigen_directory_files_to_install}
#   DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen COMPONENT Devel
#   )

# install(DIRECTORY src DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen COMPONENT Devel FILES_MATCHING PATTERN "*.h")

add_subdirectory(src)

# kinda hacky not-full file listing (but it doesnt really matter unless you want to edit eigen from an ide)
add_library(eigen STATIC dummy.cpp ${Eigen_directory_files_to_install})

target_include_directories(eigen INTERFACE
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
)

