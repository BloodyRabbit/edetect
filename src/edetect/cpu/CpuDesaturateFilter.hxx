/** @file
 * @brief Declaration of class CpuDesaturateFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef CPU__CPU_DESATURATE_FILTER_HXX__INCL__
#define CPU__CPU_DESATURATE_FILTER_HXX__INCL__

#include "filters/IDesaturateFilter.hxx"

/**
 * @brief A CPU-backed desaturation filter.
 *
 * @author Jan Bobek
 */
class CpuDesaturateFilter
: public IDesaturateFilter
{
protected:
    /// @copydoc IDesaturateFilter::desaturateAverage(IImage&, const IImage&)
    void desaturateAverage( IImage& dest, const IImage& src );
    /// @copydoc IDesaturateFilter::desaturateLightness(IImage&, const IImage&)
    void desaturateLightness( IImage& dest, const IImage& src );
    /// @copydoc IDesaturateFilter::desaturateLuminosity(IImage&, const IImage&)
    void desaturateLuminosity( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_DESATURATE_FILTER_HXX__INCL__ */
