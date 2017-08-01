#include "models.h"
#include <Eigen/Dense>

using namespace std;


void Homography::warp(const ParameterStruct& params, const PointVec& pts,
                      PointVec& wpts)
{
  Transform T = paramsToTransform(params);
  for (int i = 0; i < pts.size(); i++) {
    wpts[i] = (T.block<2, 2>(0, 0)*pts[i] + T.block<2, 1>(0, 2)) /
              (T.block<1, 2>(2, 0)*pts[i] + 1);
  }
}
void Homography::compose_warp(const ParameterStruct& p, const ParameterStruct& dp,
                              ParameterStruct& new_p)
{
  Transform Hp = paramsToTransform(p);
  Transform Hdp = paramsToTransform(dp);
  Transform Hnew = Hdp.colPivHouseholderQr().solve(Hp);
  Hnew /= Hnew(2, 2);
  new_p = transformToParams(Hnew, p.cpts);
}

Homography::WarpJacobian Homography::warp_jacobian(int x, int y,
                                                   const ParameterStruct& p) 
{
  Eigen::Matrix<float, 8, 8> A;
  A <<
    p.cpts(0), p.cpts(1), 1, 0, 0, 0, -p.cpts(0) * p.cpts(0), -p.cpts(0) * p.cpts(1),
    0, 0, 0, p.cpts(0), p.cpts(1), 1, -p.cpts(1) * p.cpts(0), -p.cpts(1) * p.cpts(1),
    p.cpts(2), p.cpts(3), 1, 0, 0, 0, -p.cpts(2) * p.cpts(2), -p.cpts(2) * p.cpts(3),
    0, 0, 0, p.cpts(2), p.cpts(3), 1, -p.cpts(3) * p.cpts(2), -p.cpts(3) * p.cpts(3),
    p.cpts(4), p.cpts(5), 1, 0, 0, 0, -p.cpts(4) * p.cpts(4), -p.cpts(4) * p.cpts(5),
    0, 0, 0, p.cpts(4), p.cpts(5), 1, -p.cpts(5) * p.cpts(4), -p.cpts(5) * p.cpts(5),
    p.cpts(6), p.cpts(7), 1, 0, 0, 0, -p.cpts(6) * p.cpts(6), -p.cpts(6) * p.cpts(7),
    0, 0, 0, p.cpts(6), p.cpts(7), 1, -p.cpts(7) * p.cpts(6), -p.cpts(7) * p.cpts(7);

  Eigen::Matrix<float, 8, 2> dWdh_t;
  dWdh_t <<
    x, 0,
    y, 0,
    1, 0,
    0, x,
    0, y,
    0, 1,
    -x*x, -x*y,
    -y*x, -y*y;
  
  WarpJacobian wJ = A.transpose().colPivHouseholderQr().solve(dWdh_t).transpose();
  return wJ;
}

void Homography::inv_params(const ParameterStruct& p, ParameterStruct &dst) {
  Transform T = paramsToTransform(p);
  Transform Tinv = T.inverse();
  dst = transformToParams(Tinv, p.cpts);
}
void Homography::inc_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst) 
{
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
  dst = transformToParams(newT, p.cpts * scale);
}

void Homography::dec_params(const ParameterStruct& p, int num_levels,
                            ParameterStruct& dst) 
{
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
  dst = transformToParams(newT, p.cpts / scale);
}

Homography::ParameterStruct Homography::scale_params(const ParameterStruct& p, float scale) 
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

Homography::Transform Homography::paramsToTransform(const ParameterStruct p)
{
  ControlPoints tpts = p.vec + p.cpts;
  Eigen::Matrix<float, 8, 8> A;
  A <<
    p.cpts(0), p.cpts(1), 1, 0, 0, 0, -tpts(0) * p.cpts(0), -tpts(0) * p.cpts(1),
    0, 0, 0, p.cpts(0), p.cpts(1), 1, -tpts(1) * p.cpts(0), -tpts(1) * p.cpts(1),
    p.cpts(2), p.cpts(3), 1, 0, 0, 0, -tpts(2) * p.cpts(2), -tpts(2) * p.cpts(3),
    0, 0, 0, p.cpts(2), p.cpts(3), 1, -tpts(3) * p.cpts(2), -tpts(3) * p.cpts(3),
    p.cpts(4), p.cpts(5), 1, 0, 0, 0, -tpts(4) * p.cpts(4), -tpts(4) * p.cpts(5),
    0, 0, 0, p.cpts(4), p.cpts(5), 1, -tpts(5) * p.cpts(4), -tpts(5) * p.cpts(5),
    p.cpts(6), p.cpts(7), 1, 0, 0, 0, -tpts(6) * p.cpts(6), -tpts(6) * p.cpts(7),
    0, 0, 0, p.cpts(6), p.cpts(7), 1, -tpts(7) * p.cpts(6), -tpts(7) * p.cpts(7);
  
  Eigen::Matrix<float, 8, 1> h = A.colPivHouseholderQr().solve(tpts);

  Transform H;
  H <<
    h(0), h(1), h(2),
    h(3), h(4), h(5),
    h(6), h(7), 1;

  return H;
}
Homography::ParameterStruct Homography::transformToParams(const Transform T,
                                                         const ControlPoints cpts) 
{
  ParameterStruct p;
  for (int i = 0; i < 4; i++) {
    p.vec.segment<2>(2*i) = (T.block<2, 2>(0, 0)*cpts.segment<2>(2*i) + T.block<2, 1>(0, 2)) /
                            (T.block<1, 2>(2, 0)*cpts.segment<2>(2*i) + 1);

  }
  p.vec -= cpts;
  p.cpts = cpts;
  return p;
}

  
