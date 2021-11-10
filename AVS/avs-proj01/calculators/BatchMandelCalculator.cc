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
	data = ((int *) _mm_malloc(height * width * sizeof(int), 64));
	for (int i = 0; i < height * width; i++) {
		data[i] = limit;
	}
    realVal = (float*)_mm_malloc(blockSize * sizeof(float), 64);
    imagVal = (float*)_mm_malloc(blockSize * sizeof(float), 64);
	isSet = (bool*)malloc(blockSize * sizeof(bool));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	data = NULL;
	_mm_free(realVal);
	realVal = NULL;
	_mm_free(imagVal);
	imagVal = NULL;
	free(isSet);
	isSet = NULL;
}

int * BatchMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	int pixelsSet = 0;
	float *pReal = realVal;
	float *pImag = imagVal;
	for (int blockHeight = 0; blockHeight < height / blockSize; blockHeight++) {
		for (int blockWidht = 0; blockWidht < width / blockSize; blockWidht++) {
			for (int i = 0; i < blockSize; i++) {
				const int iGlobal = blockHeight * blockSize + i;
				int *pdata = data + (iGlobal * width + blockWidht * blockSize);
				float imag = y_start + iGlobal * dy;
				for (int j = 0; j < blockSize; j++) {
					int jGlobal = blockWidht * blockSize + j;
					pReal[j] = x_start + jGlobal * dx;
					pImag[j] = imag;
					isSet[j] = false;
				}
				for (int iteration = 0; iteration < limit && pixelsSet < blockSize; iteration++) {
					int someIndex = blockWidht * blockSize;
					#pragma omp simd reduction(+:pixelsSet) aligned(pdata : 64, pReal : 64, pImag : 64) simdlen(128)
					for (int j = 0; j < blockSize; j++) {
						float real = x_start + (someIndex + j) * dx;
						float r2 = pReal[j] * pReal[j];
						float i2 = pImag[j] * pImag[j];
						if (r2 + i2 > 4.0f && !isSet[j]) {
							*(pdata) = iteration;
							isSet[j] = true;
							pixelsSet++;
						}
						pImag[j] = 2.0f * pReal[j] * pImag[j] + imag;
						pReal[j] = r2 - i2 + real;	
						pdata++;
					}
					pdata -= blockSize;
				}
				pixelsSet = 0;
			}
		}
	}
	return data;
}