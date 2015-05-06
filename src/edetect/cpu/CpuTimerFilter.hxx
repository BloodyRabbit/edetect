/** @file
 * @brief Declaration of class CpuTimerFilter.
 *
 * @author Jan Bobek
 * @since 1st May 2015
 */

#ifndef CPU__CPU_TIMER_FILTER_HXX__INCL__
#define CPU__CPU_TIMER_FILTER_HXX__INCL__

#include "filters/ITimerFilter.hxx"

/**
 * @brief CPU-backed implementation of a timer filter.
 *
 * @author Jan Bobek
 */
class CpuTimerFilter
: public ITimerFilter
{
public:
    /// @copydoc ITimerFilter::ITimerFilter(IImageFilter*)
    CpuTimerFilter( IImageFilter* filter = NULL );

    /// @copydoc ITimerFilter::filter(IImage&)
    void filter( IImage& image );
};

#endif /* !CPU__CPU_TIMER_FILTER_HXX__INCL__ */
