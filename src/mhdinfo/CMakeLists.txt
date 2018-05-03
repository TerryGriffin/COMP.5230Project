cmake_minimum_required(VERSION 2.8.12)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif()

project(mhdinfo)

find_package(ITK)
include(${ITK_USE_FILE})
find_package( OpenCV REQUIRED )
include_directories( ../include ${OpenCV_INCLUDE_DIRS} )

file(GLOB Lib_Sources RELATIVE "../lib" "*.cpp")
add_executable(mhdinfo mhdinfo.cpp ${Lib_Sources})

target_link_libraries(mhdinfo ${ITK_LIBRARIES})
target_link_libraries( mhdinfo ${OpenCV_LIBS} )
