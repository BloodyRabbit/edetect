/** @file
 * @brief Declaration of CudaMarrHildrethOperatorFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_MARR_HILDRETH_OPERATOR_FILTER_HXX__INCL__
#define CUDA__CUDA_MARR_HILDRETH_OPERATOR_FILTER_HXX__INCL__

#include "IMarrHildrethOperatorFilter.hxx"
#include "cuda/CudaConvolution2dFilter.hxx"
#include "cuda/CudaZeroCrossFilter.hxx"

/**
 * @brief Applies Marr-Hildreth operator to the image.
 *
 * @author Jan Bobek
 */
class CudaMarrHildrethOperatorFilter
: public IMarrHildrethOperatorFilter<
    CudaConvolution2dFilter,
    CudaZeroCrossFilter >
{
protected:
    /// @copydoc IMarrHildrethOperatorFilter< F >::mergeEdges(IImage&, const IImage&)
    void mergeEdges( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_MARR_HILDRETH_OPERATOR_FILTER_HXX__INCL__ */
