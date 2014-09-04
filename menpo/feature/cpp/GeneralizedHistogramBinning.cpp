#include "HistogramBinning.h"

HistogramBinning::HistogramBinning(unsigned int windowHeight,
                                   unsigned int windowWidth,
                                   unsigned int numberOfChannels,
                                   unsigned int numberOfOrientationBins,
                                   unsigned int cellHeightAndWidthInPixels,
                                   unsigned int blockHeightAndWidthInCells,
                                   bool enableSignedGradients) {
    unsigned int descriptorLengthPerBlock = 0,
                 numberOfBlocksPerWindowVertically = 0,
                 numberOfBlocksPerWindowHorizontally = 0;

    descriptorLengthPerBlock = blockHeightAndWidthInCells *
                               blockHeightAndWidthInCells *
                               numberOfOrientationBins;
    numberOfBlocksPerWindowVertically = 1 +
    (windowHeight - blockHeightAndWidthInCells*cellHeightAndWidthInPixels)
    / cellHeightAndWidthInPixels;
    numberOfBlocksPerWindowHorizontally = 1 +
    (windowWidth - blockHeightAndWidthInCells * cellHeightAndWidthInPixels)
    / cellHeightAndWidthInPixels;

    this->numberOfOrientationBins = numberOfOrientationBins;
    this->cellHeightAndWidthInPixels = cellHeightAndWidthInPixels;
    this->blockHeightAndWidthInCells = blockHeightAndWidthInCells;
    this->enableSignedGradients = enableSignedGradients;
    this->numberOfBlocksPerWindowHorizontally =
                    numberOfBlocksPerWindowHorizontally;
    this->numberOfBlocksPerWindowVertically =
                    numberOfBlocksPerWindowVertically;
    this->descriptorLengthPerBlock = descriptorLengthPerBlock;
    this->descriptorLengthPerWindow = numberOfBlocksPerWindowHorizontally *
                                      numberOfBlocksPerWindowVertically *
                                      descriptorLengthPerBlock;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;
}

HistogramBinning::~HistogramBinning() {
}


void HistogramBinning::apply(double *windowImage, double *descriptorVector) {
    CreateHistogram(windowImage, this->numberOfOrientationBins,
                    this->cellHeightAndWidthInPixels,
                    this->blockHeightAndWidthInCells,
                    this->enableSignedGradients,
                    this->windowHeight, this->windowWidth,
                    this->numberOfChannels, descriptorVector);
}

void CreateHistogram(double *inputImage, unsigned int numberOfOrientationBins,
                     unsigned int cellHeightAndWidthInPixels,
                     unsigned int blockHeightAndWidthInCells,
                     bool signedOrUnsignedGradientsBool,
                     unsigned int imageHeight, unsigned int imageWidth,
                     unsigned int numberOfChannels, double *descriptorVector) {

    numberOfOrientationBins = (int)numberOfOrientationBins;
    cellHeightAndWidthInPixels = (double)cellHeightAndWidthInPixels;
    blockHeightAndWidthInCells = (int)blockHeightAndWidthInCells;

    unsigned int signedOrUnsignedGradients;

    if (signedOrUnsignedGradientsBool) {
        signedOrUnsignedGradients = 1;
    } else {
        signedOrUnsignedGradients = 0;
    }

    int hist1 = 2 + (imageHeight / cellHeightAndWidthInPixels);
    int hist2 = 2 + (imageWidth / cellHeightAndWidthInPixels);

    double binsSize = (1 + (signedOrUnsignedGradients == 1)) *
                      pi / numberOfOrientationBins;

    float *dx = new float[numberOfChannels];
    float *dy = new float[numberOfChannels];
    float gradientOrientation, gradientMagnitude, tempMagnitude,
          Xc, Yc, Oc, blockNorm;
    int x1 = 0, x2 = 0, y1 = 0, y2 = 0, bin1 = 0, descriptorIndex = 0;
    unsigned int x, y, i, j, k, bin2;

    vector<vector<vector<double> > > h(hist1, vector<vector<double> >
                                       (hist2, vector<double>
                                        (numberOfOrientationBins, 0.0 ) ) );
    vector<vector<vector<double> > > block(blockHeightAndWidthInCells, vector<vector<double> >
                                           (blockHeightAndWidthInCells, vector<double>
                                            (numberOfOrientationBins, 0.0) ) );

    //Calculate gradients (zero padding)
    for(unsigned int y = 0; y < imageHeight; y++) {
        for(unsigned int x = 0; x < imageWidth; x++) {
            if (x == 0) {
                for (unsigned int z = 0; z < numberOfChannels; z++)
                    dx[z] = inputImage[y + (x + 1) * imageHeight +
                                       z * imageHeight * imageWidth];
            }
            else {
                if (x == imageWidth - 1) {
                    for (unsigned int z = 0; z < numberOfChannels; z++)
                        dx[z] = -inputImage[y + (x - 1) * imageHeight +
                                            z * imageHeight * imageWidth];
                }
                else {
                    for (unsigned int z = 0; z < numberOfChannels; z++)
                        dx[z] = inputImage[y + (x + 1) * imageHeight +
                                           z * imageHeight * imageWidth] -
                                inputImage[y + (x - 1) * imageHeight +
                                           z * imageHeight * imageWidth];
                }
            }

            if(y == 0) {
                for (unsigned int z = 0; z < numberOfChannels; z++)
                    dy[z] = -inputImage[y + 1 + x * imageHeight +
                                        z * imageHeight * imageWidth];
            }
            else {
                if (y == imageHeight - 1) {
                    for (unsigned int z = 0; z < numberOfChannels; z++)
                        dy[z] = inputImage[y - 1 + x * imageHeight +
                                           z * imageHeight * imageWidth];
                }
                else {
                    for (unsigned int z = 0; z < numberOfChannels; z++)
                        dy[z] = -inputImage[y + 1 + x * imageHeight +
                                            z * imageHeight * imageWidth] +
                                 inputImage[y - 1 + x * imageHeight +
                                            z * imageHeight * imageWidth];
                }
            }

            // choose dominant channel based on magnitude
            gradientMagnitude = sqrt(dx[0] * dx[0] + dy[0] * dy[0]);
            gradientOrientation= atan2(dy[0], dx[0]);
            if (numberOfChannels > 1) {
                tempMagnitude = gradientMagnitude;
                for (unsigned int cli = 1; cli < numberOfChannels; ++cli) {
                    tempMagnitude= sqrt(dx[cli] * dx[cli] + dy[cli] * dy[cli]);
                    if (tempMagnitude > gradientMagnitude) {
                        gradientMagnitude = tempMagnitude;
                        gradientOrientation = atan2(dy[cli], dx[cli]);
                    }
                }
            }

            if (gradientOrientation < 0)
                gradientOrientation += pi +
                                       (signedOrUnsignedGradients == 1) * pi;

            // trilinear interpolation
            bin1 = (gradientOrientation / binsSize) - 1;
            bin2 = bin1 + 1;
            x1   = x / cellHeightAndWidthInPixels;
            x2   = x1 + 1;
            y1   = y / cellHeightAndWidthInPixels;
            y2   = y1 + 1;

            Xc = (x1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
            Yc = (y1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
            Oc = (bin1 + 1 + 1 - 1.5) * binsSize;

            if (bin2 == numberOfOrientationBins)
                bin2 = 0;

            if (bin1 < 0)
                bin1 = numberOfOrientationBins - 1;

            h[y1][x1][bin1] = h[y1][x1][bin1] + gradientMagnitude *
                              (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (1-((gradientOrientation-Oc)/binsSize));
            h[y1][x1][bin2] = h[y1][x1][bin2] + gradientMagnitude *
                              (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (((gradientOrientation-Oc)/binsSize));
            h[y2][x1][bin1] = h[y2][x1][bin1] + gradientMagnitude *
                              (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (1-((gradientOrientation-Oc)/binsSize));
            h[y2][x1][bin2] = h[y2][x1][bin2] + gradientMagnitude *
                              (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (((gradientOrientation-Oc)/binsSize));
            h[y1][x2][bin1] = h[y1][x2][bin1] + gradientMagnitude *
                              (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (1-((gradientOrientation-Oc)/binsSize));
            h[y1][x2][bin2] = h[y1][x2][bin2] + gradientMagnitude *
                              (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (((gradientOrientation-Oc)/binsSize));
            h[y2][x2][bin1] = h[y2][x2][bin1] + gradientMagnitude *
                              (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (1-((gradientOrientation-Oc)/binsSize));
            h[y2][x2][bin2] = h[y2][x2][bin2] + gradientMagnitude *
                              (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                              (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                              (((gradientOrientation-Oc)/binsSize));
        }
    }

    //Block normalization
    for(x = 1; x < hist2 - blockHeightAndWidthInCells; x++) {
        for (y = 1; y < hist1 - blockHeightAndWidthInCells; y++) {
            blockNorm = 0;
            for (i = 0; i < blockHeightAndWidthInCells; i++)
                for(j = 0; j < blockHeightAndWidthInCells; j++)
                    for(k = 0; k < numberOfOrientationBins; k++)
                        blockNorm += h[y+i][x+j][k] * h[y+i][x+j][k];

            blockNorm = sqrt(blockNorm);
            for (i = 0; i < blockHeightAndWidthInCells; i++) {
                for(j = 0; j < blockHeightAndWidthInCells; j++) {
                    for(k = 0; k < numberOfOrientationBins; k++) {
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
            for (i = 0; i < blockHeightAndWidthInCells; i++)
                for(j = 0; j < blockHeightAndWidthInCells; j++)
                    for(k = 0; k < numberOfOrientationBins; k++)
                        blockNorm += block[i][j][k] * block[i][j][k];

            blockNorm = sqrt(blockNorm);
            for (i = 0; i < blockHeightAndWidthInCells; i++) {
                for(j = 0; j < blockHeightAndWidthInCells; j++) {
                    for(k = 0; k < numberOfOrientationBins; k++) {
                        if (blockNorm > 0)
                            descriptorVector[descriptorIndex] =
                                block[i][j][k] / blockNorm;
                        else
                            descriptorVector[descriptorIndex] = 0.0;
                        descriptorIndex++;
                    }
                }
            }
        }
    }
    delete[] dx;
    delete[] dy;
}