/** @file
 * @brief Declaration of class LaplacianOfGaussianFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#ifndef LAPLACIAN_OF_GAUSSIAN_FILTER_HXX__INCL__
#define LAPLACIAN_OF_GAUSSIAN_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Convolves the image with
 *  the Laplacian-of-Gaussian (LoG) kernel.
 *
 * @author Jan Bobek
 */
template< typename CF >
class LaplacianOfGaussianFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     */
    LaplacianOfGaussianFilter();
    /**
     * @brief Frees the kernel.
     */
    ~LaplacianOfGaussianFilter();

    /**
     * @brief Applies the LoG kernel to the image.
     *
     * @param[in,out] image
     *   The image to apply the kernel to.
     */
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParam(const char*, const void*)
    void setParam( const char* name, const void* value );

    /**
     * @brief Sets radius of the LoG kernel.
     *
     * @param[in] radius
     *   The desired kernel radius.
     */
    void setRadius( unsigned int radius );

protected:
    /// The generated kernel.
    float* mKernel;
    /// Convolution filter we delegate to.
    CF mFilter;
};

#include "LaplacianOfGaussianFilter.txx"

#endif /* !LAPLACIAN_OF_GAUSSIAN_FILTER_HXX__INCL__ */
