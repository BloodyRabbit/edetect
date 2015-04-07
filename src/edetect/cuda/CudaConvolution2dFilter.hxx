/** @file
 * @brief Declaration of CudaConvolution2dFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_CONVOLUTION_2D_FILTER_HXX__INCL__
#define CUDA__CUDA_CONVOLUTION_2D_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Applies full 2D discrete convolution
 *   to the image.
 *
 * @author Jan Bobek
 */
class CudaConvolution2dFilter
: public IImageFilter
{
public:
    /// The largest supported radius.
    static const unsigned int MAX_RADIUS = 32;

    /**
     * @brief Initializes the filter.
     */
    CudaConvolution2dFilter();

    /**
     * @brief Performs convolution on the image.
     *
     * @param[in] image
     *   The image to convolve.
     */
    void filter( CudaImage& image );
    /// @copydoc IImageFilter::setParam(const char*, const void*)
    void setParam( const char* name, const void* value );

    /**
     * @brief Sets a new convolution kernel.
     *
     * @param[in] kernel
     *   The new kernel.
     * @param[in] radius
     *   Radius of the new kernel.
     */
    void setKernel( const float* kernel, unsigned int radius );

protected:
    /// The convolution kernel.
    const float* mKernel;
    /// Radius of the kernel.
    unsigned int mRadius;
};

#endif /* !CUDA__CUDA_CONVOLUTION_2D_FILTER_HXX__INCL__ */
