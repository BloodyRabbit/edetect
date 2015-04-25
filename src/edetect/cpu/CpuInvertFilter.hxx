/** @file
 * @brief Declaration of class CpuInvertFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#ifndef CPU__CPU_INVERT_FILTER_HXX__INCL__
#define CPU__CPU_INVERT_FILTER_HXX__INCL__

#include "filters/IInvertFilter.hxx"

/**
 * @brief CPU-backed implementation of
 *   an inversion filter.
 *
 * @author Jan Bobek
 */
class CpuInvertFilter
: public IInvertFilter
{
protected:
    /// @copydoc IInvertFilter::invert(IImage&)
    void invert( IImage& image );
};

#endif /* !CPU__CPU_INVERT_FILTER_HXX__INCL__ */
