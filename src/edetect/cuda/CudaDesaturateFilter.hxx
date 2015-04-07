/** @file
 * @brief Declaration of CudaDesaturateFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__
#define CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Desaturates the image (converts
 *   it to grayscale).
 *
 * @author Jan Bobek
 */
class CudaDesaturateFilter
: public IImageFilter
{
public:
    /**
     * @brief Desaturates the image.
     *
     * @param[in,out] image
     *   The image to desaturate.
     */
    void filter( CudaImage& image );
    /// @copydoc IImageFilter::setParam(const char*, const void*)
    void setParam( const char* name, const void* value );

protected:
    /// The chosen method of desaturation.
    enum
    {
        METHOD_AVERAGE,    ///< The Average method.
        METHOD_LIGHTNESS,  ///< The Lightness method.
        METHOD_LUMINOSITY, ///< The Luminosity method.
    } mMethod;
};

#endif /* !CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__ */
