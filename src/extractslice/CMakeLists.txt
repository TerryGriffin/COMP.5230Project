cmake_minimum_required(VERSION 2.8.12)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif()

project(extractslice)

find_package(ITK)
include(${ITK_USE_FILE})
find_package( OpenCV REQUIRED )
include_directories( ../include ${OpenCV_INCLUDE_DIRS} )

file(GLOB Lib_Sources RELATIVE "../lib" "*.cpp")
add_executable(extractslice extractslice.cpp ${Lib_Sources})

target_link_libraries(extractslice ${ITK_LIBRARIES})
target_link_libraries( extractslice ${OpenCV_LIBS} )
