/** @file
 * @brief Declaration of CudaKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA_KIRSCH_OPERATOR_FILTER_HXX__INCL__
#define CUDA_KIRSCH_OPERATOR_FILTER_HXX__INCL__

#include "CudaConvolution2dFilter.hxx"

/**
 * @brief Applies Kirsch operator to the image.
 *
 * @author Jan Bobek
 */
class CudaKirschOperatorFilter
: public CudaFilter
{
public:
    /**
     * @brief Initializes the Kirsch operator.
     */
    CudaKirschOperatorFilter();

    /**
     * @brief Applies the Kirsch operator to an image.
     *
     * @param[in,out] image
     *   The image to apply the operator to.
     */
    void process( CudaImage& image );

protected:
    /// Number of kernels used by Kirsch operator.
    static const unsigned int KERNEL_COUNT = 8;
    /// Radius of kernels used by Kirsch operator.
    static const unsigned int KERNEL_RADIUS = 1;
    /// The kernels used by Kirsch operator.
    static const float KERNELS[KERNEL_COUNT][(2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1)];

    /// The filters we use for each kernel.
    CudaConvolution2dFilter mFilters[KERNEL_COUNT];
};

#endif /* !CUDA_KIRSCH_OPERATOR_FILTER_HXX__INCL__ */
