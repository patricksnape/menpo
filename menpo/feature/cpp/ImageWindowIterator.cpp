#include "ImageWindowIterator.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>

ImageWindowIterator::ImageWindowIterator(double* _image,
    Py_ssize_t _imageHeight,
    Py_ssize_t _imageWidth,
    Py_ssize_t _numberOfChannels,
	Py_ssize_t _windowHeight,
	Py_ssize_t _windowWidth,
	Py_ssize_t _windowStepHorizontal,
	Py_ssize_t _windowStepVertical,
	bool _enablePadding) :
	    imageHeight(_imageHeight), imageWidth(_imageWidth),
	    numberOfChannels(_numberOfChannels), windowHeight(_windowHeight),
	    windowWidth(_windowWidth), windowStepHorizontal(_windowStepHorizontal),
	    windowStepVertical(_windowStepVertical), enablePadding(_enablePadding),
	    image(_image) {

    Py_ssize_t _numberOfWindowsHorizontally = 0,
               _numberOfWindowsVertically = 0;

    // Find number of windows
    if (!enablePadding) {
        _numberOfWindowsHorizontally = 1 + (imageWidth - windowWidth) / windowStepHorizontal;
        _numberOfWindowsVertically = 1 + (imageHeight - windowHeight) / windowStepVertical;
    }
    else {
        _numberOfWindowsHorizontally = 1 + ((imageWidth - 1) / windowStepHorizontal);
        _numberOfWindowsVertically = 1 + ((imageHeight - 1) / windowStepVertical);
    }
	this->numberOfWindowsHorizontally = _numberOfWindowsHorizontally;
	this->numberOfWindowsVertically = _numberOfWindowsVertically;
}

ImageWindowIterator::~ImageWindowIterator() {
}


void ImageWindowIterator::apply(double *outputImage, Py_ssize_t *windowsCenters, WindowFeature *windowFeature) {
	Py_ssize_t rowCenter, rowFrom, rowTo, columnCenter, columnFrom, columnTo,
	           i, j, k, d,
	           windowIndexHorizontal, windowIndexVertical;

    // Initialize temporary matrices
	double* windowImage = new double[windowHeight * windowWidth * numberOfChannels];
	double* descriptorVector = new double[windowFeature->descriptorLengthPerWindow];

    // Main loop
    for (windowIndexVertical = 0; windowIndexVertical < numberOfWindowsVertically; windowIndexVertical++) {
        for (windowIndexHorizontal = 0; windowIndexHorizontal < numberOfWindowsHorizontally; windowIndexHorizontal++) {
            // Find window limits
            if (!enablePadding) {
                rowFrom = windowIndexVertical * windowStepVertical;
                rowTo = rowFrom + windowHeight - 1;
                rowCenter = rowFrom + (Py_ssize_t)round((double)windowHeight / 2.0) - 1;
                columnFrom = windowIndexHorizontal * windowStepHorizontal;
                columnTo = columnFrom + windowWidth - 1;
                columnCenter = columnFrom + (Py_ssize_t)round((double)windowWidth / 2.0) - 1;
            }
            else {
                rowCenter = windowIndexVertical * windowStepVertical;
                rowFrom = rowCenter - (Py_ssize_t)round((double)windowHeight / 2.0) + 1;
                rowTo = rowFrom + windowHeight - 1;
                columnCenter = windowIndexHorizontal * windowStepHorizontal;
                columnFrom = columnCenter - (Py_ssize_t)ceil((double)windowWidth / 2.0) + 1;
                columnTo = columnFrom + windowWidth - 1;
            }

            // Copy window image
			for (i = rowFrom; i <= rowTo; i++) {
				for (j = columnFrom; j <= columnTo; j++) {
					if (i < 0 || i > imageHeight-1 || j < 0 || j > imageWidth-1)
						for (k = 0; k < numberOfChannels; k++)
							windowImage[(i-rowFrom)+windowHeight*((j-columnFrom)+windowWidth*k)] = 0;
					else
						for (k=0; k < numberOfChannels; k++)
							windowImage[(i-rowFrom)+windowHeight*((j-columnFrom)+windowWidth*k)] = image[i+imageHeight*(j+imageWidth*k)];
				}
			}

            // Compute descriptor of window
            windowFeature->apply(windowImage, descriptorVector);

            // Store results
            for (d = 0; d < windowFeature->descriptorLengthPerWindow; d++)
            	outputImage[windowIndexVertical+numberOfWindowsVertically*(windowIndexHorizontal+numberOfWindowsHorizontally*d)] = descriptorVector[d];
            windowsCenters[windowIndexVertical+numberOfWindowsVertically*windowIndexHorizontal] = rowCenter;
            windowsCenters[windowIndexVertical+numberOfWindowsVertically*(windowIndexHorizontal+numberOfWindowsHorizontally)] = columnCenter;
        }
    }

    // Free temporary matrices
    delete[] windowImage;
    delete[] descriptorVector;
}

