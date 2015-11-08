#include "ClassDescriptor.h"
#include <iostream>

bool ClassDescriptor::sum(std::vector<float> xes)
{
	float sum = 0;
	for (size_t i = 0; i < xes.size(); i++) {
		sum += xes[i];
	}
	return sum != 0;
}

void ClassDescriptor::addPixel(std::vector<float>& pixelFeatures)
{
	for (size_t i = 0; i < pixelFeatures.size(); i++) {
		numPerFeature[i]++;
		Descriptor.at<float>(i, 0) += pixelFeatures[i];
		if (classID != 0 && sum(pixelFeatures))
			std::cout << "";

	}
}

void ClassDescriptor::CalculateDescriptor()
{
	for (size_t i = 0; i < num_features_t_; i++) {
		float before = Descriptor.at<float>(i, 0);
		float after = before / numPerFeature[i];
		Descriptor.at<float>(i, 0) = after;
	}
}