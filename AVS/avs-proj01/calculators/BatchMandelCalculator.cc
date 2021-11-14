/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	// @TODO allocate & prefill memory
	data = (int *) _mm_malloc(height * width * sizeof(int), 64);
	for (int i = 0; i < height * width; i++) {
		data[i] = limit;
	}
    realVal = (float*)_mm_malloc(blockSize * blockSize * sizeof(float), 64);
    imagVal = (float*)_mm_malloc(blockSize * blockSize * sizeof(float), 64);
	pixelSet = (bool*)_mm_malloc(blockSize * blockSize * sizeof(bool), 64);
}

BatchMandelCalculator::~BatchMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	data = NULL;
	_mm_free(realVal);
	realVal = NULL;
	_mm_free(imagVal);
	imagVal = NULL;
	_mm_free(pixelSet);
	pixelSet = NULL;
}

int * BatchMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	int pixelsSet = 0;
	float *pReal = realVal;
	float *pImag = imagVal;
	bool *isSet = pixelSet;
	for (int blockHeight = 0; blockHeight < height / blockSize; blockHeight++) {
		for (int blockWidht = 0; blockWidht < width / blockSize; blockWidht++) {
			for (int iteration = 0; iteration < limit && pixelsSet < blockSize * blockSize; iteration++) {
				for (int tileHeight = 0; tileHeight < blockSize; tileHeight++) {
					const int iGlobal = blockHeight * blockSize + tileHeight;
					int *pdata = data + (iGlobal * width + blockWidht * blockSize);
					float imag = y_start + iGlobal * dy;
					float *pReal = realVal + tileHeight * blockSize;
					float *pImag = imagVal + tileHeight * blockSize;
					bool *isSet = pixelSet + tileHeight * blockSize;
					int jGlobal = blockWidht * blockSize;
					#pragma omp simd reduction(+:pixelsSet) aligned(pdata:64, pReal:64, pImag:64, isSet:64) simdlen(256)
					for (int tileWidth = 0; tileWidth < blockSize; tileWidth++) {
						float real = x_start + (jGlobal + tileWidth) * dx;
						if (iteration == 0) {
							pReal[tileWidth] = x_start + (blockWidht * blockSize + tileWidth) * dx;
							pImag[tileWidth] = imag;
							isSet[tileWidth] = false;
						}

						float r2 = pReal[tileWidth] * pReal[tileWidth];
						float i2 = pImag[tileWidth] * pImag[tileWidth];
						if (r2 + i2 > 4.0f && !isSet[tileWidth]) {
							pdata[tileWidth] = iteration;
							isSet[tileWidth] = true;
							pixelsSet++;
						}
						pImag[tileWidth] = 2.0f * pReal[tileWidth] * pImag[tileWidth] + imag;
						pReal[tileWidth] = r2 - i2 + real;		
					}
				}
			}
			pixelsSet = 0;
		}
	}
	return data;
}