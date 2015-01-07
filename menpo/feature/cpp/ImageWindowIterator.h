#pragma once
#include "WindowFeature.h"

class ImageWindowIterator {
public:
	size_t imageHeight, imageWidth,
	           numberOfChannels, windowHeight, windowWidth,
               windowStepHorizontal, windowStepVertical,
               numberOfWindowsHorizontally, numberOfWindowsVertically;
    bool enablePadding;
	ImageWindowIterator(double* _image,
	                    size_t _imageHeight,
	                    size_t _imageWidth,
	                    size_t _numberOfChannels,
	                    size_t _windowHeight,
	                    size_t _windowWidth,
	                    size_t _windowStepHorizontal,
			            size_t _windowStepVertical,
			            bool _enablePadding);
	virtual ~ImageWindowIterator();
	void apply(double *outputImage, size_t *windowsCenters,
	           WindowFeature *windowFeature);
private:
	double* image;
};
