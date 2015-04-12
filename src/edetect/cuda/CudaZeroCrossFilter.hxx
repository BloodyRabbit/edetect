/** @file
 * @brief Declaration of class CudaZeroCrossFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#ifndef CUDA__CUDA_ZERO_CROSS_FILTER_HXX__INCL__
#define CUDA__CUDA_ZERO_CROSS_FILTER_HXX__INCL__

#include "IZeroCrossFilter.hxx"

/**
 * @brief CUDA-backed zero-crossing
 *   detection filter.
 *
 * @author Jan Bobek
 */
class CudaZeroCrossFilter
: public IZeroCrossFilter
{
protected:
    /// @copydoc IZeroCrossFilter::detectZeroCross(IImage&, const IImage&)
    void detectZeroCross( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_ZERO_CROSS_FILTER_HXX__INCL__ */
