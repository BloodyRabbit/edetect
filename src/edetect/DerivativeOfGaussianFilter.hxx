/** @file
 * @brief Declaration of class DerivativeOfGaussianFilter.
 *
 * @author Jan Bobek
 * @since 18th April 2015
 */

#ifndef DERIVATIVE_OF_GAUSSIAN_FILTER_HXX__INCL__
#define DERIVATIVE_OF_GAUSSIAN_FILTER_HXX__INCL__

#include "IGeneratedKernelFilter.hxx"

/**
 * @brief Convolves the image with
 *  the derivative of Gaussian kernel.
 *
 * @author Jan Bobek
 */
template< typename CF >
class DerivativeOfGaussianFilter
: public IGeneratedKernelFilter< CF >
{
protected:
    /// @copydoc IGeneratedKernelFilter< CF >::generateKernel(unsigned int, unsigned int&)
    float* generateKernel( unsigned int radius, unsigned int& length );
};

#include "DerivativeOfGaussianFilter.txx"

#endif /* !DERIVATIVE_OF_GAUSSIAN_FILTER_HXX__INCL__ */
