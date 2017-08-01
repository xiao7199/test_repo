#include "models.h"

void Translation::warp(const ParameterStruct& params, const PointVec& pts, PointVec& wpts) {
  for (int i = 0; i < pts.size(); i++) {
    wpts[i] = pts[i] + params.vec;
  }
}
void Translation::compose_warp(const ParameterStruct& p, const ParameterStruct& dp,
                         ParameterStruct& new_p) {
  new_p.vec = p.vec - dp.vec;
}
Translation::WarpJacobian Translation::warp_jacobian(int x, int y, const ParameterStruct& p) {
  WarpJacobian wj;
  wj << 1, 0, 0, 1;
  return wj;
}
void Translation::inc_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst) {
  dst = p;
  dst.vec *= 1 << num_levels;
}
void Translation::dec_params(const ParameterStruct& p, int num_levels, ParameterStruct& dst) {
  dst = p;
  dst.vec /= 1 << num_levels;
}

Translation::ParameterStruct Translation::scale_params(const ParameterStruct& p,
                                                              float scale) {
  ParameterStruct np = p;
  np.vec *= scale;
  return np;
}
