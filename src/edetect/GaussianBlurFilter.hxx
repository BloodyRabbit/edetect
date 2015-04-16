/** @file
 * @brief Declaration of class GaussianBlurFilter.
 *
 * @author Jan Bobek
 */

#ifndef GAUSSIAN_BLUR_FILTER_HXX__INCL__
#define GAUSSIAN_BLUR_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Applies Gaussian blur to the image.
 *
 * @author Jan Bobek
 */
template< typename SCF >
class GaussianBlurFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     */
    GaussianBlurFilter();
    /**
     * @brief Releases the generated kernels.
     */
    ~GaussianBlurFilter();

    /**
     * @brief Applies the Gaussian kernel to the image.
     *
     * @param[in,out] image
     *   The image to apply the blur to.
     */
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParam(const char*, const void*)
    void setParam( const char* name, const void* value );

    /**
     * @brief Sets radius of the Gaussian kernel.
     */
    void setRadius( unsigned int radius );

protected:
    /// The generated kernel.
    float* mKernel;
    /// The convolution filter we are delegating to.
    SCF mFilter;
};

#include "GaussianBlurFilter.txx"

#endif /* !GAUSSIAN_BLUR_FILTER_HXX__INCL__ */
