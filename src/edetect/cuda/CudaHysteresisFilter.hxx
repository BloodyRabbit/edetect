/** @file
 * @brief Declaration of class CudaHysteresisFilter.
 *
 * @author Jan Bobek
 * @since 26th April 2015
 */

#ifndef CUDA__CUDA_HYSTERESIS_FILTER_HXX__INCL__
#define CUDA__CUDA_HYSTERESIS_FILTER_HXX__INCL__

#include "filters/IHysteresisFilter.hxx"

/**
 * @brief CUDA-backed implementation of
 *   a hysteresis filter.
 *
 * @author Jan Bobek
 */
class CudaHysteresisFilter
: public IHysteresisFilter
{
protected:
    /// @copydoc IHysteresisFilter::hysteresis(IImage&, const IImage&)
    void hysteresis( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_HYSTERESIS_FILTER_HXX__INCL__ */
