#ifndef BP_UTILS
#define BP_UTILS
#include <opencv2/opencv.hpp>
#include <string>
#include "types.h"

template<class M>
void warp_image(const cv::Mat& I, typename M::ParameterStruct p, cv::Mat& dst);

void pt_pyrDown(const PointVec& pts, cv::Size s, PointVec& dst);

template<class M1, class M2>
Result<M2> compose_results(Result<M1> r1, Result<M2> r2);

template<class M>
void write_params(typename M::ParameterStruct p, const char * filename);

template<class M>
bool read_params(const char * filename, typename M::ParameterStruct& dst);

void write_debug_seq(std::vector<cv::Mat>& seq, const char * dirname);

void hcat_img(const cv::Mat& a, const cv::Mat& b, cv::Mat& d);
void vcat_img(const cv::Mat& a, const cv::Mat& b, cv::Mat& d);

#endif


