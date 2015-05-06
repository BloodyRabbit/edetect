/** @file
 * @brief Declaration of class CudaDualInputTimerFilter.
 *
 * @author Jan Bobek
 * @since 6th May 2015
 */

#ifndef CUDA__CUDA_DUAL_INPUT_TIMER_FILTER_HXX__INCL__
#define CUDA__CUDA_DUAL_INPUT_TIMER_FILTER_HXX__INCL__

#include "filters/IDualInputTimerFilter.hxx"

/**
 * @brief CUDA-backed implementation of a
 *   dual-input timer filter.
 *
 * @author Jan Bobek
 */
class CudaDualInputTimerFilter
: public IDualInputTimerFilter
{
public:
    /// @copydoc IDualInputTimerFilter::IDualInputTimerFilter(IDualInputFilter*)
    CudaDualInputTimerFilter( IDualInputFilter* filter = NULL );
    /// @copydoc IDualInputTimerFilter::~IDualInputTimerFilter()
    ~CudaDualInputTimerFilter();

protected:
    /// @copydoc IDualInputTimerFilter::filter2(IImage&, const IImage&)
    void filter2( IImage& dest, const IImage& src );

    /// The CUDA events used for measurements.
    cudaEvent_t mStart, mStop;
};

#endif /* !CUDA__CUDA_DUAL_INPUT_TIMER_FILTER_HXX__INCL__ */
