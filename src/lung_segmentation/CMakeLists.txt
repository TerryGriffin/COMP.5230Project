cmake_minimum_required(VERSION 2.8.12)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif()

project(lung_segmentation)

find_package(ITK)
include(${ITK_USE_FILE})
find_package( OpenCV REQUIRED )
include_directories( ../include ${OpenCV_INCLUDE_DIRS} )

set (CMAKE_CXX_STANDARD 14)

file(GLOB Lib_Sources RELATIVE "../lib" "*.cpp")
add_executable(lung_segmentation lung_segmentation.cpp ${Lib_Sources})

target_link_libraries(lung_segmentation ${ITK_LIBRARIES})
target_link_libraries( lung_segmentation ${OpenCV_LIBS} )
