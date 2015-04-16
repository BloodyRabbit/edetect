/** @file
 * @brief Declaration of class GaussianBlurFilter.
 *
 * @author Jan Bobek
 */

#ifndef GAUSSIAN_BLUR_FILTER_HXX__INCL__
#define GAUSSIAN_BLUR_FILTER_HXX__INCL__

#include "IGeneratedKernelFilter.hxx"
#include "SeparableConvolutionFilter.hxx"

/**
 * @brief Applies Gaussian blur to the image.
 *
 * @author Jan Bobek
 */
template< typename RCF, typename CCF >
class GaussianBlurFilter
: public IGeneratedKernelFilter<
    SeparableConvolutionFilter<
        RCF, CCF > >
{
public:
    /**
     * @brief Sets radius of the Gaussian kernel.
     */
    void setRadius( unsigned int radius );

protected:
    // Improves readability
    using IGeneratedKernelFilter<
        SeparableConvolutionFilter<
            RCF, CCF > >::mKernel;
    using IGeneratedKernelFilter<
        SeparableConvolutionFilter<
            RCF, CCF > >::mFilter;
};

#include "GaussianBlurFilter.txx"

#endif /* !GAUSSIAN_BLUR_FILTER_HXX__INCL__ */
