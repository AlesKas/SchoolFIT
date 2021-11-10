/**
 * @file LineMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <malloc.h>

#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	// @TODO allocate & prefill memory
	data = (int*)_mm_malloc(width * height * sizeof(int), 64);
	for (int i = 0; i < height * width; i++) {
		data[i] = limit;
	}
    realVal = (float*)_mm_malloc(width * sizeof(float), 64);
    imagVal = (float*)_mm_malloc(width * sizeof(float), 64);
}

LineMandelCalculator::~LineMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	data = NULL;
	_mm_free(realVal);
	realVal = NULL;
	_mm_free(imagVal);
	imagVal = NULL;
}

int * LineMandelCalculator::calculateMandelbrot () {
	int pixelsSet = 0;
	float *pReal = realVal;
	float *pImag = imagVal;
	for (int heightIndex = 0; heightIndex < height; heightIndex++) {
		int *pdata = data + heightIndex * width;
		float imag = y_start + heightIndex * dy;
		for (int i = 0; i < width; i++) {
			pReal[i] = x_start + i * dx;
			pImag[i] = imag;
		}
		for (int iteration = 0; pixelsSet < width && iteration < limit; iteration++) {
			#pragma omp simd reduction(+:pixelsSet) aligned(pdata : 64, pReal : 64, pImag : 64) simdlen(128)
			for (int widthIndex = 0; widthIndex < width; widthIndex++) {
				float real = x_start + widthIndex * dx;
				float r2 = pReal[widthIndex] * pReal[widthIndex];
				float i2 = pImag[widthIndex] * pImag[widthIndex];

				if (r2 + i2 > 4.0f && pdata[widthIndex] == limit) {
					pdata[widthIndex] = iteration;
					pixelsSet++;
				}
				pImag[widthIndex] = 2.0f * pReal[widthIndex] * pImag[widthIndex] + imag;
				pReal[widthIndex] = r2 - i2 + real;
			}
		}
		pixelsSet = 0;
	}
	return data;
}
