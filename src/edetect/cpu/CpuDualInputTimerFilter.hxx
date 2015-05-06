/** @file
 * @brief Declaration of class CpuDualInputTimerFilter.
 *
 * @author Jan Bobek
 * @since 6th May 2015
 */

#ifndef CPU__CPU_DUAL_INPUT_TIMER_FILTER_HXX__INCL__
#define CPU__CPU_DUAL_INPUT_TIMER_FILTER_HXX__INCL__

#include "filters/IDualInputTimerFilter.hxx"

/**
 * @brief CPU-backed implementation of a
 *   dual-input timer filter.
 *
 * @author Jan Bobek
 */
class CpuDualInputTimerFilter
: public IDualInputTimerFilter
{
public:
    /// @copydoc IDualInputTimerFilter::IDualInputTimerFilter(IDualInputFilter*)
    CpuDualInputTimerFilter( IDualInputFilter* filter = NULL );

protected:
    /// @copydoc IDualInputTimerFilter::filter2(IImage&, const IImage&)
    void filter2( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_DUAL_INPUT_TIMER_FILTER_HXX__INCL__ */
