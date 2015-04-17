/** @file
 * @brief Declaration of class GaussianBlurFilter.
 *
 * @author Jan Bobek
 */

#ifndef GAUSSIAN_BLUR_FILTER_HXX__INCL__
#define GAUSSIAN_BLUR_FILTER_HXX__INCL__

#include "IGeneratedKernelFilter.hxx"

/**
 * @brief Applies Gaussian blur to the image.
 *
 * @author Jan Bobek
 */
template< typename CF >
class GaussianBlurFilter
: public IGeneratedKernelFilter< CF >
{
protected:
    /// @copydoc IGeneratedKernelFilter< CF >::generateKernel( radius )
    float* generateKernel( unsigned int radius );
};

#include "GaussianBlurFilter.txx"

#endif /* !GAUSSIAN_BLUR_FILTER_HXX__INCL__ */
