#include "channels.h"
void TwoBit::compute(const cv::Mat& I, cv::Mat& dst, const PointVec& pts) {

  dst.create(I.rows, I.cols, MAT_TYPE);
  int I_stride = I.cols;
  auto channel = [=](const double* p, int c) {
    double u = upper*p[c], l = lower*p[c];
    return (channel_t)(((p[c-1-I_stride] > u) ? 2 : ((p[c-1-I_stride] > l) ? 1 : 0)) << 0  |
                       ((p[c-1         ] > u) ? 2 : ((p[c-1         ] > l) ? 1 : 0)) << 2  |
                       ((p[c-1+I_stride] > u) ? 2 : ((p[c-1+I_stride] > l) ? 1 : 0)) << 4  |
                       ((p[c  +I_stride] > u) ? 2 : ((p[c  +I_stride] > l) ? 1 : 0)) << 6  |
                       ((p[c  -I_stride] > u) ? 2 : ((p[c  -I_stride] > l) ? 1 : 0)) << 8  |
                       ((p[c+1-I_stride] > u) ? 2 : ((p[c+1-I_stride] > l) ? 1 : 0)) << 10 |
                       ((p[c+1         ] > u) ? 2 : ((p[c+1         ] > l) ? 1 : 0)) << 12 |
                       ((p[c+1+I_stride] > u) ? 2 : ((p[c+1+I_stride] > l) ? 1 : 0)) << 14 );
                         
  };
  for (int i = I.cols; i < pts.size() - I.cols; i++) {
    const double* Ir = I.ptr<double>(pts[i](1));
    dst.at<channel_t>(pts[i](1), pts[i](0)) = channel(Ir, pts[i](0));
  }
}
void TwoBit::makeVisualization(cv::Mat& ch, cv::Mat& dst) {
  auto channel = [=](const channel_t* p, int c, int b) {
    return (p[c] & (0b11 << PLANE_WIDTH*b)) >> PLANE_WIDTH*b;
  };
  dst.create(ch.rows*4, ch.cols*2, CV_8UC3);
  for (int r = 1; r < ch.rows - 1; r++) {
    const channel_t* chr = ch.ptr<channel_t>(r);
    for (int c = 1; c < ch.cols - 1; c++) {
      for (int b = 0; b < 8; b++) {
        uint8_t v = 127*channel(chr, c, b);
        dst.at<cv::Vec3b>((b/2) * ch.rows + r, (b%2)*ch.cols + c) = cv::Vec3b(v, v, v);
      }
    }
  }
}

void TwoBit::makeResidualVisualization(cv::Mat& ch1, cv::Mat& ch2, cv::Mat& dst) {
  auto channel = [=](const channel_t* p, int c, int b) {
    return (p[c] & (0b11 << PLANE_WIDTH*b)) >> PLANE_WIDTH*b;
  };
  dst.create(ch1.rows*4, ch1.cols*2, CV_8UC3);
  for (int r = 1; r < ch1.rows - 1; r++) {
    const channel_t* ch1r = ch1.ptr<channel_t>(r);    
    const channel_t* ch2r = ch2.ptr<channel_t>(r);
    for (int c = 1; c < ch1.cols - 1; c++) {
      for (int b = 0; b < 8; b++) {
        int v1 = channel(ch1r, c, b);
        int v2 = channel(ch2r, c, b);
        uint8_t v = 63*((v1 - v2) + 2);
        dst.at<cv::Vec3b>((b/2) * ch1.rows + r, (b%2)*ch1.cols + c) = cv::Vec3b(v, v, v);
      }
    }
  }
}


void OneBit::compute(const cv::Mat& I, cv::Mat& dst, const PointVec& pts) {
  dst.create(I.rows, I.cols, MAT_TYPE);
  int I_stride = I.cols;
  auto channel = [=](const double* p, int c) {
    return (channel_t)((p[c-1-I_stride] > p[c]) << 0 |
                       (p[c-1         ] > p[c]) << 1 |
                       (p[c-1+I_stride] > p[c]) << 2 |
                       (p[c  +I_stride] > p[c]) << 3 |
                       (p[c  -I_stride] > p[c]) << 4 |
                       (p[c+1-I_stride] > p[c]) << 5 |
                       (p[c+1         ] > p[c]) << 6 |
                       (p[c+1+I_stride] > p[c]) << 7 );
                         
  };
  for (int i = 0; i < pts.size(); i++) {
    const double* Ir = I.ptr<double>(pts[i](1));
    dst.at<channel_t>(pts[i](1), pts[i](0)) = channel(Ir, pts[i](0));
  }  
}
void OneBit::makeVisualization(cv::Mat& ch, cv::Mat& dst) {
  auto channel = [=](const channel_t* p, int c, int b) {
    return (p[c] & (BIT_MASK << PLANE_WIDTH*b)) >> PLANE_WIDTH*b;
  };
  dst.create(ch.rows*4, ch.cols*2, CV_8UC1);
  for (int r = 1; r < ch.rows - 1; r++) {
    const channel_t* chr = ch.ptr<channel_t>(r);
    for (int c = 1; c < ch.cols - 1; c++) {
      for (int b = 0; b < 8; b++) {
        dst.at<uint8_t>((b/2) * ch.rows + r, (b%2)*ch.cols + c) = 255*channel(chr, c, b);
      }
    }
  }
}
void OneBit::makeResidualVisualization(cv::Mat& ch1, cv::Mat& ch2, cv::Mat& dst) {
  auto channel = [=](const channel_t* p, int c, int b) {
    return (p[c] & (0b1 << PLANE_WIDTH*b)) >> PLANE_WIDTH*b;
  };
  dst.create(ch1.rows*4, ch1.cols*2, CV_8UC3);
  for (int r = 1; r < ch1.rows - 1; r++) {
    const channel_t* ch1r = ch1.ptr<channel_t>(r);    
    const channel_t* ch2r = ch2.ptr<channel_t>(r);
    for (int c = 1; c < ch1.cols - 1; c++) {
      for (int b = 0; b < 8; b++) {
        int v1 = channel(ch1r, c, b);
        int v2 = channel(ch2r, c, b);
        uint8_t v = 127*((v1 - v2) + 1);
        dst.at<cv::Vec3b>((b/2) * ch1.rows + r, (b%2)*ch1.cols + c) = cv::Vec3b(v, v, v);
      }
    }
  }
}
