#ifndef BP_CHANNELS_H
#define BP_CHANNELS_H

#include "types.h"
#include <opencv2/opencv.hpp>

class TwoBit {
private:
  static constexpr double lower = 0.95, upper = 1.05;
public:
  typedef uint16_t channel_t;
  static constexpr auto MAT_TYPE = CV_16UC1;
  static constexpr int DEPTH = 8;
  static constexpr int PLANE_WIDTH = 2;
  static constexpr int BIT_MASK = 3;  
  static void makeVisualization(cv::Mat& ch, cv::Mat& dst);
  static void makeResidualVisualization(cv::Mat& ch1, cv::Mat& ch2, cv::Mat& dst);
  static void compute(const cv::Mat& I, cv::Mat& dst, const PointVec& pts);
};

class OneBit {
public:
  typedef uint8_t channel_t;
  static constexpr auto MAT_TYPE = CV_8UC1;
  static constexpr int DEPTH = 8;
  static constexpr int PLANE_WIDTH = 1;
  static constexpr int BIT_MASK = 1;
  static void makeVisualization(cv::Mat& ch, cv::Mat& dst);  
  static void makeResidualVisualization(cv::Mat& ch1, cv::Mat& ch2, cv::Mat& dst);
  static void compute(const cv::Mat& I, cv::Mat& dst, const PointVec& pts);
};

#endif
