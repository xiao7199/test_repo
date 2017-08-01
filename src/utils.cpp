#include <fstream>
#include <string>
#include "utils.h"
#include "models.h"

using namespace cv;
using namespace std;

template<class M>
void warp_image(const cv::Mat& I, typename M::ParameterStruct p, cv::Mat& dst)
{
  cv::Mat xmap(I.size(), CV_32FC1), ymap(I.size(), CV_32FC1);
  dst.create(I.size(), I.type());

  PointVec pts, wpts;
  int nr = I.rows, nc = I.cols;
  for (int r = 0; r < nr; r++) {
    for (int c = 0; c < nc; c++) {
      pts.push_back(Vector2f(c, r));
    }
  }
  wpts.resize(pts.size());

  M::warp(p, pts, wpts);
  for (int i = 0; i < wpts.size(); i++) {
    xmap.at<float>(pts[i](1), pts[i](0)) = wpts[i](0);
    ymap.at<float>(pts[i](1), pts[i](0)) = wpts[i](1);    
  }

  remap(I, dst, xmap, ymap, cv::INTER_LINEAR);
}

template void warp_image<Affine>(const cv::Mat&, typename Affine::ParameterStruct,
                                 cv::Mat&);
template void warp_image<Homography>(const cv::Mat&, typename Homography::ParameterStruct,
                                     cv::Mat&);

void pt_pyrDown(const PointVec& pts, cv::Size s, PointVec& dst) {
  dst.clear();
  cv::Mat used = cv::Mat::zeros(s.height, s.width, CV_8UC1);
  Vector2f pt;
  for (int n = 0; n < pts.size(); n++) {
    pt << (int)(pts[n](0)/2), (int)(pts[n](1)/2);
    if (used.at<uint8_t>(pt(1), pt(0)) == 0) {
      used.at<uint8_t>(pt(1), pt(0)) = 1;
      dst.push_back(pt);
    }
  }
}

template<class M1, class M2>
Result<M2> compose_results(Result<M1> r1, Result<M2> r2) {
  Result<M2> r3 = r2;
  r3.vis.clear();
  r3.iterations += r1.iterations;
  r3.vis.reserve( r1.vis.size() + r2.vis.size() );
  r3.vis.insert( r3.vis.end(), r1.vis.begin(), r1.vis.end() );
  r3.vis.insert( r3.vis.end(), r2.vis.begin(), r2.vis.end() );
  return r3;
}

template Result<Translation> compose_results(Result<Translation>,
                                             Result<Translation>);
template Result<Translation> compose_results(Result<Affine>,
                                             Result<Translation>);
template Result<Translation> compose_results(Result<Homography>,
                                             Result<Translation>);

template Result<Affine> compose_results(Result<Translation>,
                                        Result<Affine>);
template Result<Affine> compose_results(Result<Affine>,
                                        Result<Affine>);
template Result<Affine> compose_results(Result<Homography>,
                                        Result<Affine>);

template Result<Homography> compose_results(Result<Translation>,
                                            Result<Homography>);
template Result<Homography> compose_results(Result<Affine>,
                                        Result<Homography>);
template Result<Homography> compose_results(Result<Homography>,
                                            Result<Homography>);


template<class M>
void write_params(typename M::ParameterStruct p, const char * filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    file << p.vec;
  }
}
template void write_params<Translation>(typename Translation::ParameterStruct, const char *);
template void write_params<Affine>(typename Affine::ParameterStruct, const char *);
template void write_params<Homography>(typename Homography::ParameterStruct, const char *);

template<class M>
bool read_params(const char * filename, typename M::ParameterStruct& dst) {
  std::ifstream file(filename, std::ofstream::in);
  std::string line;
  if (file.is_open()) {
    for (int i = 0; i < M::DOF; i++) {
      getline(file, line);
      dst.vec(i) = stof(line);
    }
    return true;
  } 

  std::cout << "ERROR: " << filename << " not found" << std::endl;
  return false;
}

template bool read_params<Translation>(const char *, Translation::ParameterStruct&);
template bool read_params<Affine>(const char *, Affine::ParameterStruct&);
template bool read_params<Homography>(const char *, Homography::ParameterStruct&);

void write_debug_seq(std::vector<cv::Mat>& seq, const char *  dirname) {
  char filename[50];
  for (int i = 0; i < seq.size(); i++) {
    sprintf(filename, "%s/img-%04d.png", dirname, i);
    imwrite(filename, seq[i]);
  }
}

void vcat_img(const Mat& a, const Mat& b, Mat& d) {
  d.create(a.rows + b.rows, max(a.cols, b.cols), a.type());
  Mat top(d, Rect(0, 0, a.cols, a.rows));
  a.copyTo(top);
  Mat bot(d, Rect(0, a.rows, b.cols, b.rows));
  b.copyTo(bot);
}

void hcat_img(const Mat& a, const Mat& b, Mat& d) {
  d.create(max(a.rows, b.rows), a.cols + b.cols, a.type());
  Mat left(d, Rect(0, 0, a.cols, a.rows));
  a.copyTo(left);
  Mat right(d, Rect(a.cols, 0, b.cols, b.rows));
  b.copyTo(right);
}
