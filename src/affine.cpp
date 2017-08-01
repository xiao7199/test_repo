#include "models.h"
#include <Eigen/Core>
#include <Eigen/Dense>

void Affine::warp(const ParameterStruct& params, const PointVec& pts, PointVec& wpts) {
  Transform T = paramsToTransform(params);
  for (int i = 0; i < pts.size(); i++) {
    wpts[i] = (T.block<2, 2>(0, 0)*pts[i] + T.block<2, 1>(0, 2)).eval();
  }
}

void Affine::compose_warp(const ParameterStruct& p, const ParameterStruct& dp,
                                 ParameterStruct& new_p) {
  Transform Ap = paramsToTransform(p);
  Transform Adp = paramsToTransform(dp);
  Transform Anew = Adp.colPivHouseholderQr().solve(Ap);
  Anew /= Anew(2, 2);
  new_p = transformToParams(Anew, p.cpts);
}

Affine::WarpJacobian Affine::warp_jacobian(int x, int y, const ParameterStruct& p) {
  Eigen::Matrix<float, 6, 6> A;
  A <<
    p.cpts(0), p.cpts(1), 1, 0, 0, 0,
    0, 0, 0, p.cpts(0), p.cpts(1), 1,
    p.cpts(2), p.cpts(3), 1, 0, 0, 0,
    0, 0, 0, p.cpts(2), p.cpts(3), 1,
    p.cpts(4), p.cpts(5), 1, 0, 0, 0,
    0, 0, 0, p.cpts(4), p.cpts(5), 1;
  Eigen::Matrix<float, 6, 2> dWda_t;
  dWda_t <<
    x, 0,
    y, 0,
    1, 0,
    0, x,
    0, y,
    0, 1;
  WarpJacobian wj = A.transpose().colPivHouseholderQr().solve(dWda_t).transpose();
  return wj;
}

void Affine::inc_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst) {
  Eigen::Matrix<float, 3, 3> S, Si;
  float scale = (float)(1<<num_levels);
  S <<
    scale, 0, 0,
    0, scale, 0,
    0, 0, 1;
  Si <<
    1/scale, 0, 0,
    0, 1/scale, 0,
    0, 0, 1;
      
  Transform T = paramsToTransform(p);
  Transform newT = S * T * Si;
  dst = transformToParams(newT, p.cpts);
}

void Affine::dec_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst) {
  Eigen::Matrix<float, 3, 3> S, Si;
  float scale = (float)(1<<num_levels);
  S <<
    scale, 0, 0,
    0, scale, 0,
    0, 0, 1;
  Si <<
    1/scale, 0, 0,
    0, 1/scale, 0,
    0, 0, 1;
      
  Transform T = paramsToTransform(p);
  Transform newT = Si * T * S;
  dst = transformToParams(newT, p.cpts);
}

Affine::ParameterStruct Affine::scale_params(const ParameterStruct& p, float scale) 
{
  Eigen::Matrix<float, 3, 3> S, Si;
  S <<
    scale, 0, 0,
    0, scale, 0,
    0, 0, 1;
  Si <<
    1/scale, 0, 0,
    0, 1/scale, 0,
    0, 0, 1;

  Transform T = paramsToTransform(p);
  Transform newT = S * T * Si;
  return transformToParams(newT, p.cpts);
}

Affine::Transform Affine::paramsToTransform(const ParameterStruct p) {
  ControlPoints tpts = p.vec + p.cpts;
  Eigen::Matrix<float, 6, 6> A;
  A <<
    p.cpts(0), p.cpts(1), 1, 0, 0, 0,
    0, 0, 0, p.cpts(0), p.cpts(1), 1,
    p.cpts(2), p.cpts(3), 1, 0, 0, 0,
    0, 0, 0, p.cpts(2), p.cpts(3), 1,
    p.cpts(4), p.cpts(5), 1, 0, 0, 0,
    0, 0, 0, p.cpts(4), p.cpts(5), 1;
  Eigen::Matrix<float, 6, 1> a = A.colPivHouseholderQr().solve(tpts);
  Transform T;
  T <<
    a(0), a(1), a(2),
    a(3), a(4), a(5),
    0, 0, 1;
  
  return T;
}

Affine::ParameterStruct Affine::transformToParams(const Transform T, const ControlPoints cpts) {
  ParameterStruct p;
  for (int i = 0; i < 3; i++) {
    p.vec.segment<2>(2*i) = T.block<2, 2>(0, 0)*cpts.segment<2>(2*i) + T.block<2, 1>(0, 2);
  }
  p.vec -= cpts;
  p.cpts = cpts;
  return p;
}

