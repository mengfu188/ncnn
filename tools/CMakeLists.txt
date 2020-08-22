cmake_minimum_required(VERSION 3.1) # for CMAKE_CXX_STANDARD
set(CMAKE_CXX_STANDARD 11)

if(MSVC)
    # warning C4018: '<': signed/unsigned mismatch
    # warning C4244: 'argument': conversion from 'uint64_t' to 'const unsigned int', possible loss of data
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4018 /wd4244 /wd4996)
endif()

add_subdirectory(caffe)
add_subdirectory(mxnet)
add_subdirectory(onnx)
add_subdirectory(darknet)
add_subdirectory(quantize)

add_executable(ncnn2mem ncnn2mem.cpp)
target_link_libraries(ncnn2mem PRIVATE ncnn)
if(NCNN_VULKAN)
    target_link_libraries(ncnn2mem PRIVATE ${Vulkan_LIBRARY})
endif()

add_executable(ncnnoptimize ncnnoptimize.cpp)
target_link_libraries(ncnnoptimize PRIVATE ncnn)
if(NCNN_VULKAN)
    target_link_libraries(ncnnoptimize PRIVATE ${Vulkan_LIBRARY})
endif()

# add all tools to a virtual project group
set_property(TARGET ncnn2mem PROPERTY FOLDER "tools")
set_property(TARGET ncnnoptimize PROPERTY FOLDER "tools")


find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
endif()

if(OpenCV_FOUND)
    add_executable(net_output net_output.cpp)
    target_compile_definitions(net_output PRIVATE -DOpenCV_VERSION_MAJOR=${OpenCV_VERSION_MAJOR})
    target_link_libraries(net_output PRIVATE ncnn ${OpenCV_LIBS})

    # add all ncnn2table and ncnn2int8 tool to a virtual project group
    set_property(TARGET net_output PROPERTY FOLDER "tools")
else()
    message(WARNING "OpenCV not found, net_output tools won't be built")
endif()