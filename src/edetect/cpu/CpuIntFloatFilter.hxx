/** @file
 * @brief Declaration of class CpuIntFloatFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef CPU__CPU_INT_FLOAT_FILTER_HXX__INCL__
#define CPU__CPU_INT_FLOAT_FILTER_HXX__INCL__

#include "filters/IIntFloatFilter.hxx"

/**
 * @brief A CPU-backed integer-pixel/float-pixel
 *   conversion filter.
 *
 * @author Jan Bobek
 */
class CpuIntFloatFilter
: public IIntFloatFilter
{
protected:
    /// @copydoc IIntFloatFilter::convertInt2Float(IImage&, const IImage&)
    void convertInt2Float( IImage& dest, const IImage& src );
    /// @copydoc IIntFloatFilter::convertFloat2Int(IImage&, const IImage&)
    void convertFloat2Int( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_INT_FLOAT_FILTER_HXX__INCL__ */
