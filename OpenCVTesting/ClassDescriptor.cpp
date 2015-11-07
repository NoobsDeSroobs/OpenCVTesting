#include "ClassDescriptor.h"

void ClassDescriptor::addPixel(std::vector<float> pixelFeatures)
{
	for (size_t i = 0; i < pixelFeatures.size(); i++) {
		numClassPixels[i]++;
		Descriptor.at<float>(i, 0) += pixelFeatures[i];
	}
}

void ClassDescriptor::CalculateDescriptor()
{
	for (size_t i = 0; i < num_features_t_; i++) {
		Descriptor.at<float>(i, 0) = Descriptor.at<float>(i, 0) / numClassPixels[i];
	}
}