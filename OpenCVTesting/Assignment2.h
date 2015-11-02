#pragma once

#include <opencv2/opencv.hpp>

void MultivariateGaussian();

float  ConfusionMatrix(const cv::Mat& GroundTruth, const cv::Mat& ClassifiedImage);