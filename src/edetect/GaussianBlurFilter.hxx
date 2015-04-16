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
template< typename SCF >
class GaussianBlurFilter
: public IGeneratedKernelFilter< SCF >
{
public:
    /**
     * @brief Sets radius of the Gaussian kernel.
     */
    void setRadius( unsigned int radius );

protected:
    // Improves readability
    using IGeneratedKernelFilter< SCF >::mKernel;
    using IGeneratedKernelFilter< SCF >::mFilter;
};

#include "GaussianBlurFilter.txx"

#endif /* !GAUSSIAN_BLUR_FILTER_HXX__INCL__ */
