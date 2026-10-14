// OpenVINS (ov_msckf::VioManager, unmodified) for Python: measurements in, the state out.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include <opencv2/opencv.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/sensor_data.h"

namespace py = pybind11;
using namespace ov_msckf;

static std::vector<double> to_vec(const Eigen::VectorXd& v, int a, int b) {
  return std::vector<double>(v.data() + a, v.data() + b);
}

class Openvins {
public:
  explicit Openvins(const std::string& config_path) {
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);
    VioManagerOptions params;
    params.print_and_load(parser);
    if (!parser->successful()) {
      throw std::runtime_error("YamlParser failed to load config: " + config_path);
    }
    vio_ = std::make_shared<VioManager>(params);
  }

  void feed_imu(double t, py::array_t<double> wm, py::array_t<double> am) {
    if (wm.ndim() != 1 || wm.shape(0) != 3 || am.ndim() != 1 || am.shape(0) != 3) {
      throw std::invalid_argument("feed_imu: wm and am must be length-3 arrays");
    }
    auto w = wm.unchecked<1>();
    auto a = am.unchecked<1>();
    ov_core::ImuData m;
    m.timestamp = t;
    m.wm << w(0), w(1), w(2);
    m.am << a(0), a(1), a(2);
    py::gil_scoped_release release;
    vio_->feed_measurement_imu(m);
  }

  void feed_camera(double t, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img, int cam_id) {
    auto b = img.request();
    if (b.ndim != 2) {
      throw std::invalid_argument("feed_camera expects a 2-D grayscale uint8 image");
    }
    cv::Mat gray(static_cast<int>(b.shape[0]), static_cast<int>(b.shape[1]), CV_8UC1, b.ptr);
    ov_core::CameraData cam;
    cam.timestamp = t;
    cam.sensor_ids.push_back(cam_id);
    cam.images.push_back(gray.clone());
    cam.masks.push_back(cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1));
    py::gil_scoped_release release;
    vio_->feed_measurement_camera(cam);
  }

  bool initialized() { return vio_->initialized(); }

  // Time, IMU state (JPL quaternion global to IMU, position, velocity, biases) and the camera-IMU calibration.
  py::dict get_state() {
    auto st = vio_->get_state();
    py::dict d;
    d["t"] = static_cast<double>(st->_timestamp);
    Eigen::VectorXd imu = st->_imu->value();
    d["q_GtoI"] = to_vec(imu, 0, 4);
    d["p"] = to_vec(imu, 4, 7);
    d["v"] = to_vec(imu, 7, 10);
    d["bg"] = to_vec(imu, 10, 13);
    d["ba"] = to_vec(imu, 13, 16);
    if (st->_calib_IMUtoCAM.count(0)) {
      Eigen::VectorXd e = st->_calib_IMUtoCAM.at(0)->value();
      d["q_ItoC"] = to_vec(e, 0, 4);
      d["p_IinC"] = to_vec(e, 4, 7);
    }
    return d;
  }

private:
  std::shared_ptr<VioManager> vio_;
};

PYBIND11_MODULE(openvins_ext, m) {
  m.doc() = "OpenVINS (ov_msckf::VioManager) for DAVIO";
  py::class_<Openvins>(m, "VioManager")
      .def(py::init<const std::string&>(), py::arg("config_path"))
      .def("feed_imu", &Openvins::feed_imu, py::arg("t"), py::arg("wm"), py::arg("am"))
      .def("feed_camera", &Openvins::feed_camera, py::arg("t"), py::arg("img"), py::arg("cam_id") = 0)
      .def("initialized", &Openvins::initialized)
      .def("get_state", &Openvins::get_state);
}
