#include "HONV.h"


HONV::HONV(unsigned int windowHeight,
           unsigned int windowWidth,
           unsigned int numberOfChannels,
           unsigned int numberOfOrientationBins,
           unsigned int cellHeightAndWidthInPixels,
           unsigned int blockHeightAndWidthInCells,
           double l2normClipping) {
    unsigned int descriptorLengthPerBlock = 0,
                 numberOfBlocksPerWindowVertically = 0,
                 numberOfBlocksPerWindowHorizontally = 0;

    descriptorLengthPerBlock = blockHeightAndWidthInCells *
                               blockHeightAndWidthInCells *
                               numberOfOrientationBins * numberOfOrientationBins;
    numberOfBlocksPerWindowVertically = 1 +
    (windowHeight - blockHeightAndWidthInCells*cellHeightAndWidthInPixels)
    / cellHeightAndWidthInPixels;
    numberOfBlocksPerWindowHorizontally = 1 +
    (windowWidth - blockHeightAndWidthInCells * cellHeightAndWidthInPixels)
    / cellHeightAndWidthInPixels;

    this->numberOfOrientationBins = numberOfOrientationBins;
    this->cellHeightAndWidthInPixels = cellHeightAndWidthInPixels;
    this->blockHeightAndWidthInCells = blockHeightAndWidthInCells;
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

HONV::~HONV() {
}


void HONV::apply(double *windowImage, double *descriptorVector) {
    CreateHistogram(windowImage, this->numberOfOrientationBins,
                    this->cellHeightAndWidthInPixels,
                    this->blockHeightAndWidthInCells,
                    this->l2normClipping,
                    this->windowHeight, this->windowWidth,
                    this->numberOfChannels, descriptorVector);
}

void CreateHistogram(double *inputImage, unsigned int numberOfOrientationBins,
                     unsigned int cellHeightAndWidthInPixels,
                     unsigned int blockHeightAndWidthInCells,
                     double l2normClipping,
                     unsigned int imageHeight, unsigned int imageWidth,
                     unsigned int numberOfChannels,
                     double *descriptorVector) {

    numberOfOrientationBins = (int)numberOfOrientationBins;
    cellHeightAndWidthInPixels = (double)cellHeightAndWidthInPixels;
    blockHeightAndWidthInCells = (int)blockHeightAndWidthInCells;

    int hist1 = 2 + (imageHeight / cellHeightAndWidthInPixels);
    int hist2 = 2 + (imageWidth / cellHeightAndWidthInPixels);

    int phi_bins	= numberOfOrientationBins;
	int theta_bins	= numberOfOrientationBins;
    int orient      = 1;

    double phi_bin_size		= (1+(orient==1))*P2/phi_bins;
	double theta_bin_size	= (0.5*P2)/theta_bins;

    float dx, dy, phi, theta;
    float Xc, Yc, Pc, Tc, block_norm;
	float xV, yV, pV, tV;

    int x1 = 0, x2 = 0, y1 = 0, y2 = 0, phi_bin1 = 0, theta_bin1 = 0, des_indx = 0;
    unsigned int phi_bin2, theta_bin2;


	vector<vector<vector<vector<double> > > >h(hist1, vector<vector<vector<double> > >(hist2,
		vector<vector<double> >(phi_bins, vector<double>(theta_bins, 0.0) ) ) );
	vector<vector<vector<vector<double> > > >block(blockHeightAndWidthInCells, vector<vector<vector<double> > >(blockHeightAndWidthInCells,
		vector<vector<double> >(phi_bins, vector<double>(theta_bins, 0.0) ) ) );

    //Calculate gradients (zero padding)

    for(unsigned int y=0; y<imageHeight; y++) {
        for(unsigned int x=0; x<imageWidth; x++) {

			if(x==0) dx = inputImage[y +(x+1)*imageHeight];
			else{
				if (x==imageWidth-1) dx = -inputImage[y + (x-1)*imageHeight];
				else dx = inputImage[y+(x+1)*imageHeight] - inputImage[y + (x-1)*imageHeight];
			}
			if(y==0) dy = -inputImage[y+1+x*imageHeight];
			else{
				if (y==imageHeight-1) dy = inputImage[y-1+x*imageHeight];
				else dy = -inputImage[y+1+x*imageHeight] + inputImage[y-1+x*imageHeight];
			}

            phi = atan2(dy, dx);
			theta = acos(1.0 / sqrt(dx*dx + dy*dy + 1.0));

            if (phi<0) phi += P2 + (orient==1)*P2;
            if (theta<0) theta += P2 / 2.0;

            // linear interpolation
            phi_bin1 = (phi / phi_bin_size) - 1;
            phi_bin2 = phi_bin1 + 1;
			theta_bin1 = (theta / theta_bin_size) - 1;
			theta_bin2 = theta_bin1 + 1;

            x1   = x / cellHeightAndWidthInPixels;
            x2   = x1 + 1;
            y1   = y / cellHeightAndWidthInPixels;
            y2   = y1 + 1;

            Xc = (x1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
            Yc = (y1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;

            Pc = (phi_bin1 + 1 + 1 - 1.5) * phi_bin_size;
            Tc = (theta_bin1 + 1 + 1 - 1.5) * theta_bin_size;

            if (phi_bin2 == phi_bins)
                phi_bin2 = 0;

            if (phi_bin1 < 0)
                phi_bin1 = phi_bins - 1;

			if (theta_bin2 == theta_bins)
			    theta_bin2 = 0;

			if (theta_bin1 < 0)
			    theta_bin1 = theta_bins - 1;


			xV = (x+1-Xc)/cellHeightAndWidthInPixels;
			yV = (y+1-Yc)/cellHeightAndWidthInPixels;
			pV = (phi-Pc)/phi_bin_size;
			tV = (theta-Tc)/theta_bin_size;

            h[y1][x1][phi_bin1][theta_bin1] += (1-xV)*(1-yV)*(1-pV)*(1-tV);
            h[y1][x1][phi_bin2][theta_bin1] += (1-xV)*(1-yV)*pV*(1-tV);
            h[y2][x1][phi_bin1][theta_bin1] += (1-xV)*yV*(1-pV)*(1-tV);
            h[y2][x1][phi_bin2][theta_bin1] += (1-xV)*yV*pV*(1-tV);
            h[y1][x2][phi_bin1][theta_bin1] += xV*(1-yV)*(1-pV)*(1-tV);
            h[y1][x2][phi_bin2][theta_bin1] += xV*(1-yV)*pV*(1-tV);
            h[y2][x2][phi_bin1][theta_bin1] += xV*yV*(1-pV)*(1-tV);
            h[y2][x2][phi_bin2][theta_bin1] += xV*yV*pV*(1-tV);

			h[y1][x1][phi_bin1][theta_bin2] += (1-xV)*(1-yV)*(1-pV)*tV;
			h[y1][x1][phi_bin2][theta_bin2] += (1-xV)*(1-yV)*pV*tV;
			h[y2][x1][phi_bin1][theta_bin2] += (1-xV)*yV*(1-pV)*tV;
			h[y2][x1][phi_bin2][theta_bin2] += (1-xV)*yV*pV*tV;
			h[y1][x2][phi_bin1][theta_bin2] += xV*(1-yV)*(1-pV)*tV;
			h[y1][x2][phi_bin2][theta_bin2] += xV*(1-yV)*pV*tV;
			h[y2][x2][phi_bin1][theta_bin2] += xV*yV*(1-pV)*tV;
			h[y2][x2][phi_bin2][theta_bin2] += xV*yV*pV*tV;
        }
    }

    //Block normalization
    for(unsigned int x=1; x<hist2-blockHeightAndWidthInCells; x++){
        for (unsigned int y=1; y<hist1-blockHeightAndWidthInCells; y++){

            block_norm=0;
            for (unsigned int i=0; i<blockHeightAndWidthInCells; i++){
                for(unsigned int j=0; j<blockHeightAndWidthInCells; j++){
                    for(unsigned int k=0; k<phi_bins; k++){
						for (unsigned int t=0; t<theta_bins; t++){
							block_norm+=h[y+i][x+j][k][t]*h[y+i][x+j][k][t];
						}
                    }
                }
            }

            block_norm=sqrt(block_norm);
            for (unsigned int i=0; i<blockHeightAndWidthInCells; i++){
                for(unsigned int j=0; j<blockHeightAndWidthInCells; j++){
                    for(unsigned int k=0; k<phi_bins; k++){
						for (unsigned int t=0; t<theta_bins; t++){
							if (block_norm>0){
								block[i][j][k][t]=h[y+i][x+j][k][t]/block_norm;
								if (block[i][j][k][t]>l2normClipping) block[i][j][k][t]=l2normClipping;
							}
						}
                    }
                }
            }

            block_norm=0;
            for (unsigned int i=0; i<blockHeightAndWidthInCells; i++){
                for(unsigned int j=0; j<blockHeightAndWidthInCells; j++){
                    for(unsigned int k=0; k<phi_bins; k++){
						for (unsigned int t=0; t<theta_bins; t++){
							block_norm+=block[i][j][k][t]*block[i][j][k][t];
						}
                    }
                }
            }

            block_norm=sqrt(block_norm);
            for (unsigned int i=0; i<blockHeightAndWidthInCells; i++){
                for(unsigned int j=0; j<blockHeightAndWidthInCells; j++){
                    for(unsigned int k=0; k<phi_bins; k++){
						for (unsigned int t=0; t<theta_bins; t++){
							if (block_norm>0) descriptorVector[des_indx]=block[i][j][k][t]/block_norm;
							else descriptorVector[des_indx]=0.0;
							des_indx++;
						}
                    }
                }
            }
        }
    }
}