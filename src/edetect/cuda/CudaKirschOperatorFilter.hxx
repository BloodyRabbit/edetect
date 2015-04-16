/** @file
 * @brief Declaration of CudaKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_KIRSCH_OPERATOR_FILTER_HXX__INCL__
#define CUDA__CUDA_KIRSCH_OPERATOR_FILTER_HXX__INCL__

#include "IKirschOperatorFilter.hxx"
#include "cuda/CudaConvolutionFilter.hxx"

/**
 * @brief Applies Kirsch operator to the image.
 *
 * @author Jan Bobek
 */
class CudaKirschOperatorFilter
: public IKirschOperatorFilter<
    CudaConvolutionFilter >
{
protected:
    /// @copydoc IKirschOperatorFilter< T >::computeGradient( IImage*[KERNEL_COUNT] )
    void computeGradient( IImage* images[KERNEL_COUNT] );
};

#endif /* !CUDA__CUDA_KIRSCH_OPERATOR_FILTER_HXX__INCL__ */
