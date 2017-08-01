#ifndef BP_TYPES
#define BP_TYPES
#include <Eigen/Core>
#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <functional>

typedef Eigen::Vector2f Vector2f;
template <class M>
struct EigenVecStruct {
  typedef std::vector< M, Eigen::aligned_allocator<M> > T;
};
typedef EigenVecStruct<Vector2f>::T PointVec;

struct AlgorithmParameters {
  int levels = 1;
  int iterations = 10;
  float min_disp = 1.0;
  bool *debug = NULL;  
};

template <class M>
struct Result {
  int iterations = 0;
  float final_err = 0;
  typename M::ParameterStruct params;
  std::vector<cv::Mat> vis;
};

template <class M>
struct TrackFunction {
  typedef std::function<Result<M>(const cv::Mat&, const cv::Mat&,
                                  const typename M::ParameterStruct)> T;
};

#endif
