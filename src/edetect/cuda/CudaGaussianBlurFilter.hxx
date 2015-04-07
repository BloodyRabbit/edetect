/** @file
 * @brief Declaration of CudaGaussianBlurFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_GAUSSIAN_BLUR_FILTER_HXX__INCL__
#define CUDA__CUDA_GAUSSIAN_BLUR_FILTER_HXX__INCL__

#include "cuda/CudaConvolution2dSeparableFilter.hxx"

/**
 * @brief Applies Gaussian blur to the image.
 *
 * @author Jan Bobek
 */
class CudaGaussianBlurFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     */
    CudaGaussianBlurFilter();
    /**
     * @brief Releases the generated kernels.
     */
    ~CudaGaussianBlurFilter();

    /**
     * @brief Applies the Gaussian kernel to the image.
     *
     * @param[in,out] image
     *   The image to apply the blur to.
     */
    void filter( CudaImage& image );
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
    CudaConvolution2dSeparableFilter mFilter;
};

#endif /* !CUDA__CUDA_GAUSSIAN_BLUR_FILTER_HXX__INCL__ */
