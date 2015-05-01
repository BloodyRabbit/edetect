/** @file
 * @brief Declaration of class CudaTimerFilter.
 *
 * @author Jan Bobek
 * @since 1st May 2015
 */

#ifndef CUDA__CUDA_TIMER_FILTER_HXX__INCL__
#define CUDA__CUDA_TIMER_FILTER_HXX__INCL__

#include "filters/ITimerFilter.hxx"

/**
 * @brief CUDA-backed implementation of a timer filter.
 *
 * @author Jan Bobek
 */
class CudaTimerFilter
: public ITimerFilter
{
public:
    /// @copydoc ITimerFilter::ITimerFilter(IImageFilter*)
    CudaTimerFilter( IImageFilter* filter = NULL );
    /// @copydoc ITimerFilter::~ITimerFilter()
    ~CudaTimerFilter();

    /// @copydoc ITimerFilter::filter(IImage&)
    void filter( IImage& image );

protected:
    /// The CUDA events used for measurements.
    cudaEvent_t mStart, mStop;
};

#endif /* !CUDA__CUDA_TIMER_FILTER_HXX__INCL__ */
