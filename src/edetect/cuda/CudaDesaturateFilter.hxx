/** @file
 * @brief Declaration of CudaDesaturateFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__
#define CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__

#include "filters/IDesaturateFilter.hxx"

/**
 * @brief Desaturates the image (converts
 *   it to grayscale).
 *
 * @author Jan Bobek
 */
class CudaDesaturateFilter
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

#endif /* !CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__ */
