/** @file
 * @brief Declaration of CudaSobelOperatorFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_SOBEL_OPERATOR_FILTER_HXX__INCL__
#define CUDA__CUDA_SOBEL_OPERATOR_FILTER_HXX__INCL__

#include "ISobelOperatorFilter.hxx"
#include "cuda/CudaConvolutionFilter.hxx"

/**
 * @brief Applies Sobel operator to the image.
 *
 * @author Jan Bobek
 */
class CudaSobelOperatorFilter
: public ISobelOperatorFilter<
    CudaRowConvolutionFilter,
    CudaColumnConvolutionFilter >
{
protected:
    /// @copydoc ISobelOperatorFilter< F >::computeGradient(IImage&, const IImage&)
    void computeGradient( IImage& vert, const IImage& horz );
};

#endif /* !CUDA__CUDA_SOBEL_OPERATOR_FILTER_HXX__INCL__ */
