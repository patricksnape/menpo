#pragma once
#include "WindowFeature.h"

class ImageWindowIterator {
public:
	size_t imageHeight, imageWidth,
	       numberOfChannels, windowHeight, windowWidth,
           windowStepHorizontal, windowStepVertical,
           numberOfWindowsHorizontally, numberOfWindowsVertically;
    bool enablePadding;
	ImageWindowIterator(const double* _image,
	                    const size_t _imageHeight,
	                    const size_t _imageWidth,
	                    const size_t _numberOfChannels,
	                    const size_t _windowHeight,
	                    const size_t _windowWidth,
	                    const size_t _windowStepHorizontal,
			            const size_t _windowStepVertical,
			            const bool _enablePadding);
	virtual ~ImageWindowIterator();
	void apply(WindowFeature *windowFeature,
	           double *outputImage, size_t *windowsCenters);
private:
	const double* image;
};
