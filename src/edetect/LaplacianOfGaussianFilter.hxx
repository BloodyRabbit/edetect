/** @file
 * @brief Declaration of class LaplacianOfGaussianFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#ifndef LAPLACIAN_OF_GAUSSIAN_FILTER_HXX__INCL__
#define LAPLACIAN_OF_GAUSSIAN_FILTER_HXX__INCL__

#include "IGeneratedKernelFilter.hxx"

/**
 * @brief Convolves the image with
 *  the Laplacian-of-Gaussian (LoG) kernel.
 *
 * @author Jan Bobek
 */
template< typename CF >
class LaplacianOfGaussianFilter
: public IGeneratedKernelFilter< CF >
{
protected:
    /// @copydoc IGeneratedKernelFilter< CF >::generateKernel(unsigned int, unsigned int&)
    float* generateKernel( unsigned int radius, unsigned int& length );
};

#include "LaplacianOfGaussianFilter.txx"

#endif /* !LAPLACIAN_OF_GAUSSIAN_FILTER_HXX__INCL__ */
