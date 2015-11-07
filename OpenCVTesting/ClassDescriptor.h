#pragma once
#include <vector>
#include <opencv2/core/core.hpp>

class ClassDescriptor
{
public:
	std::vector<int> numClassPixels;
	cv::Mat Descriptor;
	size_t num_features_t_;

	explicit ClassDescriptor(size_t num_features_t)
		: num_features_t_(num_features_t)
	{
		numClassPixels.resize(num_features_t_, 0);
		Descriptor = cv::Mat(num_features_t_, 1, CV_32F);
	}

	void addPixel(std::vector<float> pixelFeatures);

	void CalculateDescriptor();
	
};
