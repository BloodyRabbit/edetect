/** @file
 * @brief Declaration of CudaGaussianBlurFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA_GAUSSIAN_BLUR_HXX__INCL__
#define CUDA_GAUSSIAN_BLUR_HXX__INCL__

#include "CudaConvolution2dSeparableFilter.hxx"

/**
 * @brief Applies Gaussian blur to the image.
 *
 * @author Jan Bobek
 */
class CudaGaussianBlurFilter
: public CudaFilter
{
public:
    /**
     * @brief Initializes the filter.
     *
     * @param[in] radius
     *   Desired radius of the kernel.
     */
    CudaGaussianBlurFilter(
        unsigned int radius
        );
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
    void process( CudaImage& image );

protected:
    /// The generated kernel.
    float* mKernel;
    /// The convolution filter we are delegating to.
    CudaConvolution2dSeparableFilter mFilter;
};

#endif /* !CUDA_GAUSSIAN_BLUR_HXX__INCL__ */
