/**
 * @file LineMandelCalculator.h
 * @author FULL NAME <xkaspa48@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */

#include "BaseMandelCalculator.h"

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    // @TODO add all internal parameters
    int* data;
};