
link_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(libvodla SHARED vODLA.cpp)
target_include_directories(libvodla PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(libvodla PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(libvodla PRIVATE vodh_sdk)   