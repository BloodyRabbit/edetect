/** @file
 * @brief Declaration of CudaConvolution2dSeparableFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__
#define CUDA_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__

#include "CudaFilter.hxx"

/**
 * @brief Performs two-dimensional discrete convolution
 *   with separable kernel on the image.
 *
 * @author Jan Bobek
 */
class CudaConvolution2dSeparableFilter
: public CudaFilter
{
public:
    /// The largest supported radius.
    static const unsigned int MAX_RADIUS = 32;

    /**
     * @brief Initializes the convolution filter.
     *
     * @param[in] kernelRows
     *   The convolution kernel for rows.
     * @param[in] kernelRowsRadius
     *   Radius of the row kernel.
     * @param[in] kernelColumns
     *   The convolution kernel for columns.
     * @param[in] kernelColumnsRadius
     *   Radius of the column kernel.
     */
    CudaConvolution2dSeparableFilter(
        const float* kernelRows = NULL,
        unsigned int kernelRowsRadius = 0,
        const float* kernelColumns = NULL,
        unsigned int kernelColumnsRadius = 0
        );

    /**
     * @brief Performs convolution on the image.
     *
     * @param[in] image
     *   The image to convolve.
     */
    void process( CudaImage& image );

    /**
     * @brief Sets the row kernel.
     *
     * @param[in] kernel
     *   The kernel.
     * @param[in] radius
     *   Radius of the kernel.
     */
    void setKernelRows(
        const float* kernel,
        unsigned int radius
        );
    /**
     * @brief Sets the column kernel.
     *
     * @param[in] kernel
     *   The kernel.
     * @param[in] radius
     *   Radius of the kernel.
     */
    void setKernelColumns(
        const float* kernel,
        unsigned int radius
        );


protected:
    /// The row kernel.
    const float* mKernelRows;
    /// Radius of the row kernel.
    unsigned int mKernelRowsRadius;

    /// The column kernel.
    const float* mKernelColumns;
    /// Radius of the column kernel.
    unsigned int mKernelColumnsRadius;
};

#endif /* !CUDA_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__ */
