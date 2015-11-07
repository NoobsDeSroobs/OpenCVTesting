#include "ClassDescriptor.h"

void ClassDescriptor::addPixel(std::vector<float> pixelFeatures)
{
	for (size_t i = 0; i < pixelFeatures.size(); i++) {
		numClassPixels[i]++;
		Descriptor[i] += pixelFeatures[i];
	}
}

void ClassDescriptor::CalculateDescriptor()
{
	for (size_t i = 0; i < Descriptor.size(); i++) {
		Descriptor[i] = Descriptor[i] / numClassPixels[i];
	}
}