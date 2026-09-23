// Native OpenVINS binding. The only write path is the one-shot bootstrap `initialize`
// (feed-forward initializer -> stock initialize_with_gt); nothing else injects state.
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
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "types/Vec.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/sensor_data.h"

namespace py = pybind11;
using namespace ov_msckf;

static std::vector<double> to_vec(const Eigen::VectorXd& v, int a, int b) {
  return std::vector<double>(v.data() + a, v.data() + b);
}

static py::array_t<double> to_array(const Eigen::MatrixXd& m) {
  py::array_t<double> out({(py::ssize_t)m.rows(), (py::ssize_t)m.cols()});
  auto r = out.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < m.rows(); ++i)
    for (py::ssize_t j = 0; j < m.cols(); ++j) r(i, j) = m(i, j);
  return out;
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
    if (wm.ndim() != 1 || wm.shape(0) != 3) {
      throw std::invalid_argument("feed_imu: wm must be a length-3 1-D array");
    }
    if (am.ndim() != 1 || am.shape(0) != 3) {
      throw std::invalid_argument("feed_imu: am must be a length-3 1-D array");
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

  void feed_camera(double t,
                   py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
                   int cam_id) {
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

  // One-shot bootstrap from a feed-forward initializer: the stock initialize_with_gt sets
  // the 16-state and a fixed covariance; the covariance is then replaced by the supplied
  // sigmas [theta(3) p(3) v(3) bg(3) ba(3)]. Refuses an initialized filter (startup_time is
  // set by both the native initializer and initialize_with_gt).
  void initialize(double t, py::array_t<double> q, py::array_t<double> p, py::array_t<double> v,
                  py::array_t<double> bg, py::array_t<double> ba, py::array_t<double> sigmas) {
    if (vio_->initialized_time() >= 0) {
      throw std::runtime_error("initialize: filter already initialized");
    }
    auto vec = [](py::array_t<double>& a, int n, const char* name) {
      if (a.ndim() != 1 || a.shape(0) != n) {
        throw std::invalid_argument(std::string("initialize: ") + name + " must have length " + std::to_string(n));
      }
      auto r = a.unchecked<1>();
      Eigen::VectorXd out(n);
      for (int i = 0; i < n; ++i) out(i) = r(i);
      return out;
    };
    Eigen::VectorXd qv = vec(q, 4, "q"), pv = vec(p, 3, "p"), vv = vec(v, 3, "v"),
                    bgv = vec(bg, 3, "bg"), bav = vec(ba, 3, "ba"), sg = vec(sigmas, 15, "sigmas");
    if (qv.norm() <= 0) throw std::invalid_argument("initialize: zero quaternion");
    qv.normalize();
    if (qv(3) < 0) qv = -qv;
    for (int i = 0; i < 15; ++i) {
      if (!(sg(i) > 0)) throw std::invalid_argument("initialize: sigmas must be positive");
    }
    Eigen::Matrix<double, 17, 1> s;
    s << t, qv, pv, vv, bgv, bav;
    py::gil_scoped_release release;
    vio_->initialize_with_gt(s);
    auto st = vio_->get_state();
    std::vector<std::shared_ptr<ov_type::Type>> order = {st->_imu};
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(15, 15);
    for (int i = 0; i < 15; ++i) cov(i, i) = sg(i) * sg(i);
    StateHelper::set_initial_covariance(st, cov, order);
  }

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
    if (st->_cam_intrinsics.count(0)) {
      Eigen::VectorXd k = st->_cam_intrinsics.at(0)->value();
      d["cam_k"] = std::vector<double>(k.data(), k.data() + k.size());
    }
    // Live camera-IMU time offset (calib_cam_timeoffset), for calibration-convergence scoring.
    d["dt"] = static_cast<double>(st->_calib_dt_CAMtoIMU->value()(0));
    return d;
  }

  // Filter self-assessment: the covariance the filter actually believes it has, alongside the
  // state. Mirrors srvins_binding.cpp::get_diagnostics's cov_imu/cov_calib0 construction via
  // StateHelper::get_marginal_covariance; ov_msckf::VioManager has no last_msckf_stats()
  // equivalent (that accounting is an ov_srvins-only addition), so this omits msckf_*.
  py::dict get_diagnostics() {
    auto st = vio_->get_state();
    py::dict d;
    d["t"] = static_cast<double>(st->_timestamp);
    d["n_clones"] = static_cast<int>(st->_clones_IMU.size());
    std::vector<std::shared_ptr<ov_type::Type>> order = {st->_imu};
    Eigen::MatrixXd cov = StateHelper::get_marginal_covariance(st, order);
    // 15x15 IMU error-state covariance: [theta(3) p(3) v(3) bg(3) ba(3)].
    d["cov_imu"] = to_array(cov);

    if (st->_calib_IMUtoCAM.count(0)) {
      std::vector<std::shared_ptr<ov_type::Type>> calib_order = {st->_calib_IMUtoCAM.at(0)};
      d["cov_calib0"] = to_array(StateHelper::get_marginal_covariance(st, calib_order));
    }
    return d;
  }

private:
  std::shared_ptr<VioManager> vio_;
};

PYBIND11_MODULE(openvins_ext, m) {
  m.doc() = "Davio shim over ov_msckf::VioManager (vendored, stock OpenVINS)";
  py::class_<Openvins>(m, "VioManager")
      .def(py::init<const std::string&>(), py::arg("config_path"))
      .def("feed_imu", &Openvins::feed_imu, py::arg("t"), py::arg("wm"), py::arg("am"))
      .def("feed_camera", &Openvins::feed_camera, py::arg("t"), py::arg("img"),
           py::arg("cam_id") = 0)
      .def("get_diagnostics", &Openvins::get_diagnostics)
      .def("initialize", &Openvins::initialize, py::arg("t"), py::arg("q"), py::arg("p"),
           py::arg("v"), py::arg("bg"), py::arg("ba"), py::arg("sigmas"))
      .def("initialized", &Openvins::initialized)
      .def("get_state", &Openvins::get_state);
}
