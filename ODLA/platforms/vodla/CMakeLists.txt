link_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_odla_library(vodla SHARED vODLA.cpp)

target_include_directories(vodla PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(vodla PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vodla PRIVATE vodh_sdk)   