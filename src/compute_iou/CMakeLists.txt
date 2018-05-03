cmake_minimum_required(VERSION 2.8.12)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif()

project(compute_iou)

find_package(ITK)
include(${ITK_USE_FILE})
find_package( OpenCV REQUIRED )
include_directories( ../include ${OpenCV_INCLUDE_DIRS} )

set (CMAKE_CXX_STANDARD 14)

file(GLOB Lib_Sources RELATIVE "../lib" "*.cpp")
add_executable(compute_iou compute_iou.cpp ${Lib_Sources})

target_link_libraries(compute_iou ${ITK_LIBRARIES})
target_link_libraries( compute_iou ${OpenCV_LIBS} )
