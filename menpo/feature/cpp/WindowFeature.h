#pragma once

// If you want to use this outside of Python,
// simply typedef Py_ssize_t to long int and comment this import out
#include "Python.h"

#if defined(_MSC_VER)
    #define round(x) (((x) >= 0) ? ((x) + 0.5) : ((x) - 0.5))
#endif

class WindowFeature {
public:
	WindowFeature();
	virtual ~WindowFeature();
	virtual void apply(double *windowImage, double *descriptorVector) = 0;
	Py_ssize_t descriptorLengthPerWindow;
};
