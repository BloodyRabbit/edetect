/** @file
 * @brief Declaration of class CpuZeroCrossFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#ifndef CPU__CPU_ZERO_CROSS_FILTER_HXX__INCL__
#define CPU__CPU_ZERO_CROSS_FILTER_HXX__INCL__

#include "filters/IZeroCrossFilter.hxx"

/**
 * @brief CPU-backed zero-crossing
 *   detection filter.
 *
 * @author Jan Bobek
 */
class CpuZeroCrossFilter
: public IZeroCrossFilter
{
protected:
    /// @copydoc IZeroCrossFilter::detectZeroCross(IImage&, const IImage&)
    void detectZeroCross( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_ZERO_CROSS_FILTER_HXX__INCL__ */
