#ifndef BP_MODEL_H
#define BP_MODEL_H

#include <Eigen/Core>
#include "types.h"

class Translation {
  
public:
  static constexpr int DOF = 2;
  typedef Eigen::Matrix<float, 2, 1> ParameterVector;
  typedef Eigen::Matrix<float, Eigen::Dynamic, DOF> Jacobian;
  typedef Eigen::Matrix<float, 2, DOF> WarpJacobian;
  typedef Eigen::Matrix<float, DOF, DOF> Hessian;
  struct ParameterStruct {
    ParameterStruct() {
      vec << 0, 0;
    }
    ParameterVector vec;
  };
  static void warp(const ParameterStruct& params, const PointVec& pts, PointVec& wpts);
  static void compose_warp(const ParameterStruct& p, const ParameterStruct& dp,
                           ParameterStruct& new_p);
  static WarpJacobian warp_jacobian(int x, int y, const ParameterStruct& p);
  static void inc_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst);
  static void dec_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst);
  static ParameterStruct scale_params(const ParameterStruct& p, float scale);
};


class Affine {
public:
  static constexpr int DOF = 6;
  typedef Eigen::Matrix<float, DOF, 1> ParameterVector;
  typedef Eigen::Matrix<float, Eigen::Dynamic, DOF> Jacobian;
  typedef Eigen::Matrix<float, 2, DOF> WarpJacobian;
  typedef Eigen::Matrix<float, DOF, DOF> Hessian;
  typedef Eigen::Matrix<float, DOF, 1> ControlPoints;
  struct ParameterStruct {
    ParameterStruct() {
      vec << 0, 0, 0, 0, 0, 0;
      cpts << 0, 0, 1, 0, 0, 1;
    }
    ParameterStruct(ControlPoints _cpts) {
      vec << 0, 0, 0, 0, 0, 0;
      cpts = _cpts;
    }
    
    ParameterVector vec;
    ControlPoints cpts;
  };


  static void warp(const ParameterStruct& params, const PointVec& pts, PointVec& wpts);
  static void compose_warp(const ParameterStruct& p, const ParameterStruct& dp,
                           ParameterStruct& new_p);
  static WarpJacobian warp_jacobian(int x, int y, const ParameterStruct& p);
  static void inc_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst);
  static void dec_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst);
  static ParameterStruct scale_params(const ParameterStruct& p, float scale);
private:
  typedef Eigen::Matrix<float, 3, 3> Transform;

  static Transform paramsToTransform(const ParameterStruct p);
  static ParameterStruct transformToParams(const Transform T, const ControlPoints cpts);  
};

class Homography {
public:
  static constexpr int DOF = 8;
  typedef Eigen::Matrix<float, DOF, 1> ParameterVector;
  typedef Eigen::Matrix<float, Eigen::Dynamic, DOF> Jacobian;
  typedef Eigen::Matrix<float, 2, DOF> WarpJacobian;
  typedef Eigen::Matrix<float, DOF, DOF> Hessian;
  typedef Eigen::Matrix<float, DOF, 1> ControlPoints;
  struct ParameterStruct {
    ParameterStruct() {
      vec << 0, 0, 0, 0, 0, 0, 0, 0;
      cpts << 0, 0, 1, 0, 0, 1, 1, 1;
    }
    ParameterStruct(ControlPoints _cpts) {
      vec << 0, 0, 0, 0, 0, 0, 0, 0;
      cpts = _cpts;
    }
    
    ParameterVector vec;
    ControlPoints cpts;
  };


  static void warp(const ParameterStruct& params, const PointVec& pts, PointVec& wpts);
  static void compose_warp(const ParameterStruct& p, const ParameterStruct& dp,
                           ParameterStruct& new_p);
  static WarpJacobian warp_jacobian(int x, int y, const ParameterStruct& p);
  static void inv_params(const ParameterStruct& p, ParameterStruct& dst);
  static void inc_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst);
  static void dec_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst);
  static ParameterStruct scale_params(const ParameterStruct& p, float scale);
private:
  typedef Eigen::Matrix<float, 3, 3> Transform;

  static Transform paramsToTransform(const ParameterStruct p);
  static ParameterStruct transformToParams(const Transform T, const ControlPoints cpts);  
};
#endif



