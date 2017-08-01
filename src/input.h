#ifndef BP_INPUT_H
#define BP_INPUT_H

#include <opencv2/opencv.hpp>
#include <string>

void readMat(const std::string filename, cv::Mat& M);
void readFloatMat(const std::string filename, cv::Mat& dst);
void readMask(const std::string filename, cv::Mat& M);
#endif
