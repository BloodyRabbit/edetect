/** @file
 * @brief Declaration of class CudaInvertFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#ifndef CUDA__CUDA_INVERT_FILTER_HXX__INCL__
#define CUDA__CUDA_INVERT_FILTER_HXX__INCL__

#include "filters/IInvertFilter.hxx"

/**
 * @brief CUDA-backed implementation of
 *   an inversion filter.
 *
 * @author Jan Bobek
 */
class CudaInvertFilter
: public IInvertFilter
{
protected:
    /// @copydoc IInvertFilter::invert(IImage&)
    void invert( IImage& image );
};

#endif /* !CUDA__CUDA_INVERT_FILTER_HXX__INCL__ */
