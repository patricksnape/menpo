#pragma once
#include "WindowFeature.h"

class ImageWindowIterator {
public:
	Py_ssize_t imageHeight, imageWidth,
	           numberOfChannels, windowHeight, windowWidth,
               windowStepHorizontal, windowStepVertical,
               numberOfWindowsHorizontally, numberOfWindowsVertically;
    bool enablePadding;
	ImageWindowIterator(double* _image,
	                    Py_ssize_t _imageHeight,
	                    Py_ssize_t _imageWidth,
	                    Py_ssize_t _numberOfChannels,
	                    Py_ssize_t _windowHeight,
	                    Py_ssize_t _windowWidth,
	                    Py_ssize_t _windowStepHorizontal,
			            Py_ssize_t _windowStepVertical,
			            bool _enablePadding);
	virtual ~ImageWindowIterator();
	void apply(double *outputImage, Py_ssize_t *windowsCenters,
	           WindowFeature *windowFeature);
private:
	double* image;
};
