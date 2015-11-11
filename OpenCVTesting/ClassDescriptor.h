#pragma once
#include <vector>
#include <opencv2/core/core.hpp>

class ClassDescriptor
{
public:
	std::vector<int> numPerFeature;
	cv::Mat Descriptor;
	size_t num_features_t_;
	size_t classID;

	explicit ClassDescriptor(size_t num_features_t, size_t classID)
		: num_features_t_(num_features_t), classID(classID)
	{
		numPerFeature.resize(num_features_t_, 0);
		Descriptor = cv::Mat(num_features_t_, 1, CV_32F);
		for (size_t i = 0; i < num_features_t_; i++) {
			Descriptor.at<float>(i, 0) = 0;
		}
	}

	bool sum(std::vector<float> xes);
	void addPixel(std::vector<float>& pixelFeatures);
	void CalculateDescriptor();
	
};
