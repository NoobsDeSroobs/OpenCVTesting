#pragma once
#include <vector>
#include <opencv2/core/core.hpp>

class ClassDescriptor
{
public:
	std::vector<int> numClassPixels;
	std::vector<float> Descriptor;
	size_t num_features_t_;

	explicit ClassDescriptor(size_t num_features_t)
		: num_features_t_(num_features_t)
	{
		numClassPixels.resize(num_features_t_, 0);
		Descriptor.resize(num_features_t_, 0);;
	}

	void addPixel(std::vector<float> pixelFeatures);

	void CalculateDescriptor();
	
};
