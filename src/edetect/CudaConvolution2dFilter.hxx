/** @file
 * @brief Declaration of CudaConvolution2dFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA_CONVOLUTION_2D_FILTER_HXX__INCL__
#define CUDA_CONVOLUTION_2D_FILTER_HXX__INCL__

#include "CudaFilter.hxx"

/**
 * @brief Applies full 2D discrete convolution
 *   to the image.
 *
 * @author Jan Bobek
 */
class CudaConvolution2dFilter
: public CudaFilter
{
public:
    /// The largest supported radius.
    static const unsigned int MAX_RADIUS = 32;

    /**
     * @brief Initializes the convolution filter.
     *
     * @param[in] kernel
     *   The convolution kernel.
     * @param[in] radius
     *   Radius of the kernel.
     */
    CudaConvolution2dFilter(
        const float* kernel = NULL,
        unsigned int radius = 0
        );

    /**
     * @brief Performs convolution on the image.
     *
     * @param[in] image
     *   The image to convolve.
     */
    void process( CudaImage& image );

    /**
     * @brief Sets the kernel to use.
     *
     * @param[in] kernel
     *   The convolution kernel.
     * @param[in] radius
     *   Radius of the kernel.
     */
    void setKernel(
        const float* kernel,
        unsigned int radius
        );

protected:
    /// The convolution kernel.
    const float* mKernel;
    /// Radius of the kernel.
    unsigned int mRadius;
};

#endif /* !CUDA_CONVOLUTION_2D_FILTER_HXX__INCL__ */
