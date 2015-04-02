/** @file
 * @brief Declaration of CudaSobelOperatorFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_SOBEL_OPERATOR_FILTER_HXX__INCL__
#define CUDA__CUDA_SOBEL_OPERATOR_FILTER_HXX__INCL__

#include "cuda/CudaConvolution2dSeparableFilter.hxx"

/**
 * @brief Applies Sobel operator to the image.
 *
 * @author Jan Bobek
 */
class CudaSobelOperatorFilter
: public CudaFilter
{
public:
    /**
     * @brief Initializes the filter.
     */
    CudaSobelOperatorFilter();

    /**
     * @brief Applies the Sobel operator to the image.
     *
     * @param[in,out] image
     *   The image to apply the operator to.
     */
    void process( CudaImage& image );

protected:
    /// The filter we use for vertical edges.
    CudaConvolution2dSeparableFilter mVertFilter;
    /// The filter we use for horizontal edges.
    CudaConvolution2dSeparableFilter mHorzFilter;

    /// Radius of both kernels.
    static const unsigned int KERNEL_RADIUS = 1;
    /// The (-1 0 1) component of Sobel kernel.
    static const float KERNEL_1_0_1[2 * KERNEL_RADIUS + 1];
    /// The (1 2 1) component of Sobel kernel.
    static const float KERNEL_1_2_1[2 * KERNEL_RADIUS + 1];
};

#endif /* !CUDA__CUDA_SOBEL_OPERATOR_FILTER_HXX__INCL__ */
