#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "tracker.h"
#include "models.h"
#include "channels.h"
#include "utils.h"

using namespace cv;
using namespace std;

template <class M, class C>
Tracker<M, C>::Tracker(AlgorithmParameters ap): _iterations(ap.iterations),
                                                _min_disp(ap.min_disp),
                                                _debug(ap.debug)
{}

template <class M, class C>
void Tracker<M, C>::set_template(const Mat& I, const PointVec& pts,
                                 const typename M::ParameterStruct& identity_params) {
  I.copyTo(_I);
  C::compute(I, _channels1, pts);
  _num_pts = pts.size();
  _pts = pts;
  _wI.create(I.size(), I.type());
  _xmap.create(I.size(), CV_32FC1);
  _ymap.create(I.size(), CV_32FC1);   
  /*******
   ** COMPUTE THE JACOBIAN
   ******/
  int stride = I.cols;
  _J.resize(C::DEPTH * _num_pts, M::DOF);
  _Jweighted.resize(C::DEPTH * _num_pts, M::DOF);  
  _residuals.resize(C::DEPTH * _num_pts, 1);
  _weights.resize(C::DEPTH * _num_pts, 1);    
  /**
   * Compute the channel x- and y- gradient at col:=x for bitplane:bp
   **/
  int m = C::BIT_MASK;  
  auto G = [&](Vector2f pt, int bp) {
    int bit = C::PLANE_WIDTH*bp;

    float ix1 = (_channels1.at<typename C::channel_t>(pt(1)  , pt(0)+1) & m<<bit) >> bit;
    float ix2 = (_channels1.at<typename C::channel_t>(pt(1)  , pt(0)-1) & m<<bit) >> bit;
    float iy1 = (_channels1.at<typename C::channel_t>(pt(1)+1, pt(0)  ) & m<<bit) >> bit;
    float iy2 = (_channels1.at<typename C::channel_t>(pt(1)-1, pt(0)  ) & m<<bit) >> bit;
    return Eigen::Matrix<float, 1, 2>(0.5f*(ix1-ix2), 0.5f*(iy1-iy2));
  };
  for (int i = 0; i < _num_pts; i++) {
    typename M::WarpJacobian Jw = M::warp_jacobian(pts[i](0), pts[i](1), identity_params);
    for (int bp = 0; bp < C::DEPTH; bp++) {
      _J.row(C::DEPTH*i+bp) = G(pts[i], bp) * Jw;
    }
  }
}

template <class M, class C>
Result<M> Tracker<M, C>::step(const cv::Mat& I, const cv::Mat& mask,
                              const typename M::ParameterStruct& init_params) {
  typename M::ParameterStruct params = init_params, dp = init_params, new_params;
  _wpts.resize(_num_pts);
  Result<M> r;
  r.vis.clear();
  int m;
  Mat templ_vis;
  if (_debug != NULL && *_debug) {
    Mat mask_int;
    mask.convertTo(mask_int, C::MAT_TYPE);
    Mat masked_channels = _channels1.mul(mask_int);
    C::makeVisualization(_channels1, templ_vis);
  }
  
  // namedWindow("Template", CV_WINDOW_AUTOSIZE);
  // namedWindow("Warped", CV_WINDOW_AUTOSIZE);
  //cout << "init_params: " << init_params.vec.transpose() << endl;
  for (m = 0; m < _iterations; m++) {
    /**
     * Warp image and mask
     **/
    M::warp(params, _pts, _wpts);
    for (int i = 0; i < _num_pts; i++) {
      _xmap.at<float>(_pts[i](1), _pts[i](0)) = _wpts[i](0);
      _ymap.at<float>(_pts[i](1), _pts[i](0)) = _wpts[i](1);
    }
    remap(I, _wI, _xmap, _ymap, INTER_LINEAR);
    remap(mask, _wmask, _xmap, _ymap, INTER_LINEAR);
    C::compute(_wI, _channels2, _pts);
    /**
     * Compute the residuals
     **/
    int mask = C::BIT_MASK;
    auto res = [&](Vector2f pt, int bp) {
      int bit = C::PLANE_WIDTH*bp;
      return (float)(((_channels2.at<typename C::channel_t>(pt(1), pt(0)) & mask<<bit) >> bit) -
                     ((_channels1.at<typename C::channel_t>(pt(1), pt(0)) & mask<<bit) >> bit));
    };
    for (int i = 0; i < _num_pts; i++) {
      for (int bp = 0; bp < C::DEPTH; bp++) {
        _residuals(C::DEPTH*i+bp) = res(_pts[i], bp);
        _weights(C::DEPTH*i+bp) = _wmask.at<uint8_t>(_pts[i](1), _pts[i](0));
      }
    }
    /**
     * Compute H and solve for the update warp
     **/
    for (int d = 0; d < M::DOF; d++) {
      _Jweighted.col(d) = _J.col(d).cwiseProduct(_weights);
    }
    _H = _Jweighted.transpose() * _Jweighted;
    dp.vec = _H.ldlt().solve(_Jweighted.transpose() * _residuals);

    M::compose_warp(params, dp, new_params);

    // cout << "Disp: " << (params.vec-new_params.vec).norm()
    //      << " Err: " << _residuals.norm()
    //      << " Params: " << new_params.vec.transpose() << endl;

    //Compute debugging information *debug is true.
    if (_debug != NULL && *_debug) {
      //namedWindow("Test", CV_WINDOW_AUTOSIZE);
      Mat warp_vis, ch_vis, all_vis, full_vis, both_imgs, both_imgs_color, both_imgs_int, diff_vis;
      // imshow("Test", I/255);
      // waitKey(0);
      // imshow("Test", _wI/255);
      // waitKey(0);

      C::makeVisualization(_channels2, warp_vis);
      diff_vis = 63*((warp_vis - templ_vis) + 2);
      hcat_img(warp_vis, templ_vis, ch_vis);
      hcat_img(ch_vis, diff_vis, all_vis);
      hcat_img(_wI, _I, both_imgs);
      both_imgs.convertTo(both_imgs, CV_32FC1);
      cvtColor(both_imgs, both_imgs_color, CV_GRAY2RGB);
      both_imgs_color.convertTo(both_imgs_int, CV_8UC3);
      vcat_img(both_imgs_int, all_vis, full_vis);
      r.vis.push_back(full_vis);
    }
    
    if ((params.vec-new_params.vec).squaredNorm() < _min_disp*_min_disp) {
      m++;
      params = new_params;      
      break;
    }
    params = new_params;

    
    
    // imshow("Template", templ_vis);
    // imshow("Warped", warp_vis);
    // cout << "waiting" << endl;
    // waitKey(0);
  } //End main iteration loop

  r.iterations = m;
  r.params = params;
  r.final_err = _residuals.norm();
  return r;
}

template class Tracker<Translation, TwoBit>;
template class Tracker<Affine, TwoBit>;
template class Tracker<Homography, TwoBit>;


template class Tracker<Translation, OneBit>;
template class Tracker<Affine, OneBit>;
template class Tracker<Homography, OneBit>;
