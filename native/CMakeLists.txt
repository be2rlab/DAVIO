cmake_minimum_required(VERSION 3.18)
project(openvins_ext LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(OV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/open_vins" CACHE PATH
    "Root of the vendored OpenVINS checkout")
set(OV_INSTALL "${OV_ROOT}/install")
set(OV_MSCKF_LIB "${OV_INSTALL}/ov_msckf/lib/libov_msckf_lib.so")
set(OV_CORE_LIB  "${OV_INSTALL}/ov_core/lib/libov_core_lib.so")
set(OV_INIT_LIB  "${OV_INSTALL}/ov_init/lib/libov_init_lib.so")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)
find_package(rclcpp REQUIRED)   # ov_core/cmake/ROS2.cmake builds the libs with ROS_AVAILABLE==2

pybind11_add_module(openvins_ext openvins_binding.cpp)

target_compile_definitions(openvins_ext PRIVATE ROS_AVAILABLE=2)

target_include_directories(openvins_ext PRIVATE
    ${OV_ROOT}/ov_msckf/src ${OV_ROOT}/ov_core/src ${OV_ROOT}/ov_init/src ${EIGEN3_INCLUDE_DIR})

target_link_libraries(openvins_ext PRIVATE ${OV_MSCKF_LIB} ${OV_CORE_LIB}
    -Wl,--no-as-needed ${OV_INIT_LIB} -Wl,--as-needed ${OpenCV_LIBS} rclcpp::rclcpp)

# So the imported module finds the colcon-built libraries without sourcing the workspace.
set_target_properties(openvins_ext PROPERTIES
    BUILD_RPATH "${OV_INSTALL}/ov_msckf/lib;${OV_INSTALL}/ov_core/lib;${OV_INSTALL}/ov_init/lib")
