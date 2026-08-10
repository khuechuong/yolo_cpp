cmake_minimum_required(VERSION 3.18)
project(yolo_culvert)

find_package(OpenCV REQUIRED)

include_directories(
    /usr/local/include
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(/usr/local/lib)

add_executable(yolo_culvert main.cpp)

target_link_libraries(yolo_culvert
    onnxruntime
    ${OpenCV_LIBS}
)

set_property(TARGET yolo_culvert PROPERTY CXX_STANDARD 17)