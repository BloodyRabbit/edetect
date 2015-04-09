/** @file
 * @brief Declaration of CudaDesaturateFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__
#define CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__

#include "IDesaturateFilter.hxx"

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
    /// @copydoc IDesaturateFilter::desaturateAverage(IImage&, const IImage&)
    void desaturateAverage( IImage& dest, const IImage& src );
    /// @copydoc IDesaturateFilter::desaturateLightness(IImage&, const IImage&)
    void desaturateLightness( IImage& dest, const IImage& src );
    /// @copydoc IDesaturateFilter::desaturateLuminosity(IImage&, const IImage&)
    void desaturateLuminosity( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__ */
