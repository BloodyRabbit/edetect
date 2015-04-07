/** @file
 * @brief Declaration of CudaConvolution2dSeparableFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__
#define CUDA__CUDA_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Performs two-dimensional discrete convolution
 *   with separable kernel on the image.
 *
 * @author Jan Bobek
 */
class CudaConvolution2dSeparableFilter
: public IImageFilter
{
public:
    /// The largest supported radius.
    static const unsigned int MAX_RADIUS = 32;

    /**
     * @brief Initializes the filter.
     */
    CudaConvolution2dSeparableFilter();

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
     * @brief Sets the row kernel.
     *
     * @param[in] kernel
     *   The row kernel.
     * @param[in] radius
     *   Radius of the row kernel.
     */
    void setRowKernel( const float* kernel, unsigned int radius );
    /**
     * @brief Sets the column kernel.
     *
     * @param[in] kernel
     *   The column kernel.
     * @param[in] radius
     *   Radius of the column kernel.
     */
    void setColumnKernel( const float* kernel, unsigned int radius );

protected:
    /// The row kernel.
    const float* mRowKernel;
    /// Radius of the row kernel.
    unsigned int mRowKernelRadius;

    /// The column kernel.
    const float* mColumnKernel;
    /// Radius of the column kernel.
    unsigned int mColumnKernelRadius;
};

#endif /* !CUDA__CUDA_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__ */
