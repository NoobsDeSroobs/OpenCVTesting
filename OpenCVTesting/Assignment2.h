#pragma once

#include <opencv2/opencv.hpp>

void MultivariateGaussian();

float  ConfusionMatrix(const cv::Mat& GroundTruth, const cv::Mat& ClassifiedImage);

void StartAssignment2();

float ComputeFeature(cv::Mat GLCM, size_t start_x_t, size_t end_x_t, size_t start_y_t, size_t end_y_t);

std::vector<float> ComputeQFeatures(cv::Mat GLCM);