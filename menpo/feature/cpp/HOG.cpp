#include "HOG.h"

HOG::HOG(const size_t windowHeight,
         const size_t windowWidth,
         const size_t numberOfChannels,
         const unsigned int method,
         const size_t numberOfOrientationBins,
         const size_t cellHeightAndWidthInPixels,
         const size_t blockHeightAndWidthInCells,
         const bool enableSignedGradients,
         const double l2normClipping) {
    size_t descriptorLengthPerBlock = 0,
           numberOfBlocksPerWindowVertically = 0,
           numberOfBlocksPerWindowHorizontally = 0;

    if (method == DALAL_TRIGGS) {
        descriptorLengthPerBlock = blockHeightAndWidthInCells *
                                   blockHeightAndWidthInCells *
                                   numberOfOrientationBins;

        numberOfBlocksPerWindowVertically = 1 +
          (windowHeight - blockHeightAndWidthInCells * cellHeightAndWidthInPixels)
          / cellHeightAndWidthInPixels;

        numberOfBlocksPerWindowHorizontally = 1 +
          (windowWidth - blockHeightAndWidthInCells * cellHeightAndWidthInPixels)
          / cellHeightAndWidthInPixels;
    }
    else if (method == ZHU_RAMANAN) {
        descriptorLengthPerBlock = 31;  // 27 + 4

        numberOfBlocksPerWindowVertically = (size_t)round((double)windowHeight /
                          (double)cellHeightAndWidthInPixels) - 2;

        numberOfBlocksPerWindowHorizontally = (size_t)round((double)windowWidth /
                          (double)cellHeightAndWidthInPixels) - 2;
    }

    this->method = method;
    this->numberOfOrientationBins = numberOfOrientationBins;
    this->cellHeightAndWidthInPixels = cellHeightAndWidthInPixels;
    this->blockHeightAndWidthInCells = blockHeightAndWidthInCells;
    this->enableSignedGradients = enableSignedGradients;
    this->l2normClipping = l2normClipping;
    this->numberOfBlocksPerWindowHorizontally = numberOfBlocksPerWindowHorizontally;
    this->numberOfBlocksPerWindowVertically = numberOfBlocksPerWindowVertically;
    this->descriptorLengthPerBlock = descriptorLengthPerBlock;
    this->descriptorLengthPerWindow = numberOfBlocksPerWindowHorizontally *
                                      numberOfBlocksPerWindowVertically *
                                      descriptorLengthPerBlock;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;
}

HOG::~HOG() {
}


void HOG::apply(const double *windowImage, double *descriptorVector) {
    if (method == DALAL_TRIGGS) {
        DalalTriggsHOGdescriptor(windowImage, this->numberOfOrientationBins,
                                 this->cellHeightAndWidthInPixels,
                                 this->blockHeightAndWidthInCells,
                                 this->enableSignedGradients,
                                 this->l2normClipping, this->windowHeight,
                                 this->windowWidth, this->numberOfChannels,
                                 descriptorVector);
    } else {
        ZhuRamananHOGdescriptor(windowImage, this->cellHeightAndWidthInPixels,
                                this->windowHeight, this->windowWidth,
                                this->numberOfChannels, descriptorVector);
    }
}


// ZHU & RAMANAN: Face Detection, Pose Estimation and Landmark Localization
//                in the Wild
void ZhuRamananHOGdescriptor(const double* inputImage,
                             const size_t cellHeightAndWidthInPixels,
                             const size_t imageHeight,
                             const size_t imageWidth,
                             const size_t numberOfChannels,
                             double *descriptorMatrix) {
    // unit vectors used to compute gradient orientation
    static const double uu[9] = {1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736,
                                 -0.5000, -0.7660, -0.9397};
    static const double vv[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848,
                                 0.8660, 0.6428, 0.3420};

    // memory for caching orientation histograms & their norms
    const size_t blocks[2] = {(size_t)round((double)imageHeight / (double)cellHeightAndWidthInPixels),
                              (size_t)round((double)imageWidth / (double)cellHeightAndWidthInPixels)};

    double* hist = (double *)calloc(blocks[0] * blocks[1] * 18, sizeof(double));
    double* norm = (double *)calloc(blocks[0] * blocks[1], sizeof(double));

    // memory for HOG features
    const size_t out[3] = {inline_max(blocks[0] - 2, 0),
                               inline_max(blocks[1] - 2, 0),
                               31};  // 27 + 4

    const size_t visible[2] = {blocks[0] * cellHeightAndWidthInPixels,
                                   blocks[1] * cellHeightAndWidthInPixels};

    for (size_t x = 1; x < visible[1] - 1; x++) {
        for (size_t y = 1; y < visible[0] - 1; y++) {
            // compute gradient
            // first channel
            const double *s = inputImage + inline_min(x, imageWidth-2) * imageHeight +
                              inline_min(y, imageHeight-2);
            double dy = *(s + 1) - *(s - 1);
            double dx = *(s + imageHeight) - *(s - imageHeight);
            double v = dx * dx + dy * dy;
            // rest of channels
            for (size_t z = 1; z < numberOfChannels; z++) {
                s += imageHeight * imageWidth;
                double dy2 = *(s + 1) - *(s - 1);
                double dx2 = *(s + imageHeight) - *(s - imageHeight);
                double v2 = dx2 * dx2 + dy2 * dy2;
                // pick channel with strongest gradient
                if (v2 > v) {
                    v = v2;
                    dx = dx2;
                    dy = dy2;
                }
            }

            // snap to one of 18 orientations
            double best_dot = 0;
            unsigned int best_o = 0;
            for (unsigned int o = 0; o < 9; o++) {
                double dot = uu[o] * dx + vv[o] * dy;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot) {
                    best_dot = - dot;
                    best_o = o + 9;
                }
            }

            // add to 4 histograms around pixel using linear interpolation
            double xp = ((double)x + 0.5) /
                        (double)cellHeightAndWidthInPixels - 0.5;
            double yp = ((double)y + 0.5) /
                        (double)cellHeightAndWidthInPixels - 0.5;
            long long ixp = (long long)floor(xp);
            long long iyp = (long long)floor(yp);
            double vx0 = xp - ixp;
            double vy0 = yp - iyp;
            double vx1 = 1.0 - vx0;
            double vy1 = 1.0 - vy0;
            v = sqrt(v);

            if (ixp >= 0 && iyp >= 0)
                *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1])
                    += vx1 * vy1 * v;

            if (ixp+1 < blocks[1] && iyp >= 0)
                *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1])
                    += vx0 * vy1 * v;

            if (ixp >= 0 && iyp+1 < blocks[0])
                *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1])
                    += vx1 * vy0 * v;

            if (ixp+1 < blocks[1] && iyp+1 < blocks[0])
                *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1])
                    += vx0 * vy0 * v;
        }
    }

    // compute energy in each block by summing over orientations
    for (unsigned int o = 0; o < 9; o++) {
        double *src1 = hist + o * blocks[0] * blocks[1];
        double *src2 = hist + (o + 9) * blocks[0] * blocks[1];
        double *dst = norm;
        double *end = norm + blocks[1] * blocks[0];
        while (dst < end) {
            *(dst++) += (*src1 + *src2) * (*src1 + *src2);
            src1++;
            src2++;
        }
    }

    // compute features
    for (size_t x = 0; x < out[1]; x++) {
        for (size_t y = 0; y < out[0]; y++) {
            double *dst = descriptorMatrix + x * out[0] + y;
            double *src, *p, n1, n2, n3, n4;

            p = norm + (x + 1) * blocks[0] + y + 1;
            n1 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + EPS);
            p = norm + (x + 1) * blocks[0] + y;
            n2 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + EPS);
            p = norm + x * blocks[0] + y + 1;
            n3 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + EPS);
            p = norm + x * blocks[0] + y;
            n4 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + EPS);

            double t1 = 0;
            double t2 = 0;
            double t3 = 0;
            double t4 = 0;

            // contrast-sensitive features
            src = hist + (x + 1) * blocks[0] + (y + 1);
            for (unsigned int o = 0; o < 18; o++) {
                double h1 = inline_min(*src * n1, 0.2);
                double h2 = inline_min(*src * n2, 0.2);
                double h3 = inline_min(*src * n3, 0.2);
                double h4 = inline_min(*src * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
                dst += out[0] * out[1];
                src += blocks[0] * blocks[1];
            }

            // contrast-insensitive features
            src = hist + (x + 1) * blocks[0] + (y + 1);
            for (unsigned int o = 0; o < 9; o++) {
                double sum = *src + *(src + 9 * blocks[0] * blocks[1]);
                double h1 = inline_min(sum * n1, 0.2);
                double h2 = inline_min(sum * n2, 0.2);
                double h3 = inline_min(sum * n3, 0.2);
                double h4 = inline_min(sum * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                dst += out[0] * out[1];
                src += blocks[0] * blocks[1];
            }

            // texture features
            *dst = 0.2357 * t1;
            dst += out[0] * out[1];
            *dst = 0.2357 * t2;
            dst += out[0] * out[1];
            *dst = 0.2357 * t3;
            dst += out[0] * out[1];
            *dst = 0.2357 * t4;
        }
    }
    free(hist);
    free(norm);
}


// DALAL & TRIGGS: Histograms of Oriented Gradients for Human Detection
void DalalTriggsHOGdescriptor(const double *inputImage,
                              const size_t numberOfOrientationBins,
                              const size_t cellHeightAndWidthInPixels,
                              const size_t blockHeightAndWidthInCells,
                              const bool signedOrUnsignedGradients,
                              const double l2normClipping, size_t imageHeight,
                              const size_t imageWidth,
                              const size_t numberOfChannels,
                              double *descriptorVector) {

    const float fCellHeightAndWidthInPixels = (float)cellHeightAndWidthInPixels;
    const size_t hist1 = 2 + (imageHeight / fCellHeightAndWidthInPixels);
    const size_t hist2 = 2 + (imageWidth / fCellHeightAndWidthInPixels);
    const double binsSize = (1 + signedOrUnsignedGradients) * pi / numberOfOrientationBins;

    float *dx = new float[numberOfChannels];
    float *dy = new float[numberOfChannels];
    size_t descriptorIndex = 0;
    float blockNorm = 0.0;

    vector<vector<vector<double> > > h(hist1, vector<vector<double> >
                                       (hist2, vector<double>
                                        (numberOfOrientationBins, 0.0 ) ) );
    vector<vector<vector<double> > > block(blockHeightAndWidthInCells, vector<vector<double> >
                                           (blockHeightAndWidthInCells, vector<double>
                                            (numberOfOrientationBins, 0.0) ) );

    //Calculate gradients (zero padding)
    for(size_t y = 0; y < imageHeight; y++) {
        for(size_t x = 0; x < imageWidth; x++) {
            if (x == 0) {
                for (size_t z = 0; z < numberOfChannels; z++)
                    dx[z] = inputImage[y + (x + 1) * imageHeight +
                                       z * imageHeight * imageWidth];
            }
            else {
                if (x == imageWidth - 1) {
                    for (size_t z = 0; z < numberOfChannels; z++)
                        dx[z] = -inputImage[y + (x - 1) * imageHeight +
                                            z * imageHeight * imageWidth];
                }
                else {
                    for (size_t z = 0; z < numberOfChannels; z++)
                        dx[z] = inputImage[y + (x + 1) * imageHeight +
                                           z * imageHeight * imageWidth] -
                                inputImage[y + (x - 1) * imageHeight +
                                           z * imageHeight * imageWidth];
                }
            }

            if(y == 0) {
                for (size_t z = 0; z < numberOfChannels; z++)
                    dy[z] = -inputImage[y + 1 + x * imageHeight +
                                        z * imageHeight * imageWidth];
            }
            else {
                if (y == imageHeight - 1) {
                    for (size_t z = 0; z < numberOfChannels; z++)
                        dy[z] = inputImage[y - 1 + x * imageHeight +
                                           z * imageHeight * imageWidth];
                }
                else {
                    for (size_t z = 0; z < numberOfChannels; z++)
                        dy[z] = -inputImage[y + 1 + x * imageHeight +
                                            z * imageHeight * imageWidth] +
                                 inputImage[y - 1 + x * imageHeight +
                                            z * imageHeight * imageWidth];
                }
            }

            // choose dominant channel based on magnitude
            float gradientMagnitude = sqrt(dx[0] * dx[0] + dy[0] * dy[0]);
            float gradientOrientation = atan2(dy[0], dx[0]);
            if (numberOfChannels > 1) {
                float tempMagnitude = gradientMagnitude;
                for (size_t cli = 1; cli < numberOfChannels; ++cli) {
                    tempMagnitude = sqrt(dx[cli] * dx[cli] + dy[cli] * dy[cli]);
                    if (tempMagnitude > gradientMagnitude) {
                        gradientMagnitude = tempMagnitude;
                        gradientOrientation = atan2(dy[cli], dx[cli]);
                    }
                }
            }

            if (gradientOrientation < 0)
                gradientOrientation += pi + signedOrUnsignedGradients * pi;

            // trilinear interpolation
            int bin1 = floor(0.5 + gradientOrientation / binsSize) - 1;
            int bin2 = bin1 + 1;
            int x1   = floor(0.5 + x / fCellHeightAndWidthInPixels);
            int x2   = x1 + 1;
            int y1   = floor(0.5 + y / fCellHeightAndWidthInPixels);
            int y2   = y1 + 1;

            int Xc = (x1 + 1 - 1.5) * fCellHeightAndWidthInPixels + 0.5;
            int Yc = (y1 + 1 - 1.5) * fCellHeightAndWidthInPixels + 0.5;
            int Oc = (bin1 + 1 + 1 - 1.5) * binsSize;

            if (bin2 == numberOfOrientationBins)
                bin2 = 0;

            if (bin1 < 0)
                bin1 = numberOfOrientationBins - 1;

            const float widthInterp = (x + 1 - Xc) / fCellHeightAndWidthInPixels;
            const float heightInterp = (y + 1 - Yc) / fCellHeightAndWidthInPixels;
            const float orientationInterp = (gradientOrientation - Oc) / binsSize;

            h[y1][x1][bin1] = h[y1][x1][bin1] +
                                gradientMagnitude *
                                (1 - widthInterp) * (1 - heightInterp) *
                                (1 - orientationInterp);
            h[y1][x1][bin2] = h[y1][x1][bin2] +
                                gradientMagnitude *
                                (1 - widthInterp) * (1 - heightInterp) *
                                orientationInterp;
            h[y2][x1][bin1] = h[y2][x1][bin1] +
                                gradientMagnitude *
                                (1 - widthInterp) * heightInterp *
                                (1 - orientationInterp);
            h[y2][x1][bin2] = h[y2][x1][bin2] +
                                gradientMagnitude *
                                (1 - widthInterp) * heightInterp *
                                orientationInterp;
            h[y1][x2][bin1] = h[y1][x2][bin1] +
                                gradientMagnitude *
                                widthInterp * (1 - heightInterp) *
                                (1 - orientationInterp);
            h[y1][x2][bin2] = h[y1][x2][bin2] +
                                gradientMagnitude *
                                widthInterp * (1 - heightInterp) *
                                orientationInterp;
            h[y2][x2][bin1] = h[y2][x2][bin1] +
                                gradientMagnitude *
                                widthInterp * heightInterp *
                                (1 - orientationInterp);
            h[y2][x2][bin2] = h[y2][x2][bin2] +
                                gradientMagnitude *
                                widthInterp * heightInterp *
                                orientationInterp;
        }
    }

    //Block normalization
    for(size_t x = 1; x < hist2 - blockHeightAndWidthInCells; x++) {
        for(size_t y = 1; y < hist1 - blockHeightAndWidthInCells; y++) {
            blockNorm = 0;
            for(size_t i = 0; i < blockHeightAndWidthInCells; i++) {
                for(size_t j = 0; j < blockHeightAndWidthInCells; j++) {
                    for(size_t k = 0; k < numberOfOrientationBins; k++) {
                        blockNorm += h[y+i][x+j][k] * h[y+i][x+j][k];
                    }
                }
            }

            blockNorm = sqrt(blockNorm);
            for (size_t i = 0; i < blockHeightAndWidthInCells; i++) {
                for(size_t j = 0; j < blockHeightAndWidthInCells; j++) {
                    for(size_t k = 0; k < numberOfOrientationBins; k++) {
                        if (blockNorm > 0) {
                            block[i][j][k] = h[y+i][x+j][k] / blockNorm;
                            if (block[i][j][k] > l2normClipping)
                                block[i][j][k] = l2normClipping;
                        }
                        else {
                            block[i][j][k] = 0;
                        }
                    }
                }
            }

            blockNorm = 0;
            for(size_t i = 0; i < blockHeightAndWidthInCells; i++) {
                for(size_t j = 0; j < blockHeightAndWidthInCells; j++) {
                    for(size_t k = 0; k < numberOfOrientationBins; k++) {
                        blockNorm += block[i][j][k] * block[i][j][k];
                    }
                }
             }

            blockNorm = sqrt(blockNorm);
            for(size_t i = 0; i < blockHeightAndWidthInCells; i++) {
                for(size_t j = 0; j < blockHeightAndWidthInCells; j++) {
                    for(size_t k = 0; k < numberOfOrientationBins; k++) {
                        if (blockNorm > 0) {
                            descriptorVector[descriptorIndex] =
                                block[i][j][k] / blockNorm;
                        } else {
                            descriptorVector[descriptorIndex] = 0.0;
                        }
                        descriptorIndex++;
                    }
                }
            }
        }
    }
    delete[] dx;
    delete[] dy;
}
