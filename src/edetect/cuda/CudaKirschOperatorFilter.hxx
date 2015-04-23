/** @file
 * @brief Declaration of CudaKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_KIRSCH_OPERATOR_FILTER_HXX__INCL__
#define CUDA__CUDA_KIRSCH_OPERATOR_FILTER_HXX__INCL__

#include "filters/IKirschOperatorFilter.hxx"

/**
 * @brief Applies Kirsch operator to the image.
 *
 * @author Jan Bobek
 */
class CudaKirschOperatorFilter
: public IKirschOperatorFilter
{
protected:
    /// @copydoc IKirschOperatorFilter::applyKirschOperator(IImage&, const IImage&)
    void applyKirschOperator( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_KIRSCH_OPERATOR_FILTER_HXX__INCL__ */
