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
    /// @copydoc IDesaturateFilter::desaturateAverageInt(IImage&, const IImage&)
    void desaturateAverageInt( IImage& dest, const IImage& src );
    /// @copydoc IDesaturateFilter::desaturateAverageFloat(IImage&, const IImage&)
    void desaturateAverageFloat( IImage& dest, const IImage& src );

    /// @copydoc IDesaturateFilter::desaturateLightnessInt(IImage&, const IImage&)
    void desaturateLightnessInt( IImage& dest, const IImage& src );
    /// @copydoc IDesaturateFilter::desaturateLightnessFloat(IImage&, const IImage&)
    void desaturateLightnessFloat( IImage& dest, const IImage& src );

    /// @copydoc IDesaturateFilter::desaturateLuminosityInt(IImage&, const IImage&)
    void desaturateLuminosityInt( IImage& dest, const IImage& src );
    /// @copydoc IDesaturateFilter::desaturateLuminosityFloat(IImage&, const IImage&)
    void desaturateLuminosityFloat( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_DESATURATE_FILTER_HXX__INCL__ */
