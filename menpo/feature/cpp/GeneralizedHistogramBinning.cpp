#include "GeneralizedHistogramBinning.h"

GeneralizedHistogramBinning::GeneralizedHistogramBinning(
                                   unsigned int windowHeight,
                                   unsigned int windowWidth,
                                   unsigned int numberOfChannels,
                                   unsigned int numberOfOrientationBins,
                                   unsigned int cellHeightAndWidthInPixels,
                                   unsigned int blockHeightAndWidthInCells,
                                   bool enableSignedGradients,
                                   double l2normClipping) {
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
    this->l2normClipping = l2normClipping;
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

GeneralizedHistogramBinning::~GeneralizedHistogramBinning() {
}


void GeneralizedHistogramBinning::apply(double *windowImage, double *descriptorVector) {
    CreateHistogram(windowImage, this->numberOfOrientationBins,
                    this->cellHeightAndWidthInPixels,
                    this->blockHeightAndWidthInCells,
                    this->enableSignedGradients, this->l2normClipping,
                    this->windowHeight, this->windowWidth,
                    this->numberOfChannels, descriptorVector);
}

void CreateHistogram(double *inputImage, unsigned int numberOfOrientationBins,
                     unsigned int cellHeightAndWidthInPixels,
                     unsigned int blockHeightAndWidthInCells,
                     bool signedOrUnsignedGradientsBool, double l2normClipping,
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
                      P / numberOfOrientationBins;

    float gradientOrientation, gradientMagnitude, Xc, Yc, Oc, blockNorm;
    int x1 = 0, x2 = 0, y1 = 0, y2 = 0, bin1 = 0, descriptorIndex = 0;
    unsigned int x, y, i, j, k, bin2;

    vector<vector<vector<double> > > h(hist1, vector<vector<double> >
                                       (hist2, vector<double>
                                        (numberOfOrientationBins, 0.0 ) ) );
    vector<vector<vector<double> > > block(blockHeightAndWidthInCells, vector<vector<double> >
                                           (blockHeightAndWidthInCells, vector<double>
                                            (numberOfOrientationBins, 0.0) ) );

    //Calculate quantity to bin
    if (numberOfChannels == 1) {
        for(unsigned int y = 0; y < imageHeight; y++) {
            for(unsigned int x = 0; x < imageWidth; x++) {
                gradientOrientation = inputImage[y + x * imageHeight];

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

                h[y1][x1][bin1] = h[y1][x1][bin1] +
                                  (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                  (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                  (1-((gradientOrientation-Oc)/binsSize));
                h[y1][x1][bin2] = h[y1][x1][bin2] +
                                  (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                  (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                  (((gradientOrientation-Oc)/binsSize));
                h[y2][x1][bin1] = h[y2][x1][bin1] +
                                  (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                  (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                  (1-((gradientOrientation-Oc)/binsSize));
                h[y2][x1][bin2] = h[y2][x1][bin2] +
                                  (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                  (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                  (((gradientOrientation-Oc)/binsSize));
                h[y1][x2][bin1] = h[y1][x2][bin1] +
                                  (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                  (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                  (1-((gradientOrientation-Oc)/binsSize));
                h[y1][x2][bin2] = h[y1][x2][bin2] +
                                  (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                  (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                  (((gradientOrientation-Oc)/binsSize));
                h[y2][x2][bin1] = h[y2][x2][bin1] +
                                  (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                  (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                  (1-((gradientOrientation-Oc)/binsSize));
                h[y2][x2][bin2] = h[y2][x2][bin2] +
                                  (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                  (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                  (((gradientOrientation-Oc)/binsSize));
            }
        }
    }
    else {
        for(unsigned int y = 0; y < imageHeight; y++) {
            for(unsigned int x = 0; x < imageWidth; x++) {
                gradientMagnitude = sqrt(inputImage[y + x * imageHeight] * inputImage[y + x * imageHeight] +
                                         inputImage[y + x * imageHeight + imageHeight * imageWidth] * inputImage[y + x * imageHeight + imageHeight * imageWidth]);
                gradientOrientation= atan2(inputImage[y + x * imageHeight + imageHeight * imageWidth], inputImage[y + x * imageHeight]);

                if (gradientOrientation < 0)
                    gradientOrientation += P +
                                           (signedOrUnsignedGradients == 1) * P;

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
}
