/** @file
 * @brief Declaration of class IDesaturateFilter.
 *
 * @author Jan Bobek
 * @since 9th April 2015
 */

#ifndef IDESATURATE_FILTER_HXX__INCL__
#define IDESATURATE_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of the desaturate
 *   (gray-scale) conversion filter.
 *
 * @author Jan Bobek
 */
class IDesaturateFilter
: public IImageFilter
{
public:
    /**
     * @brief Describes method used for desaturation.
     *
     * @author Jan Bobek
     */
    enum Method
    {
        METHOD_AVERAGE,    ///< The Average method.
        METHOD_LIGHTNESS,  ///< The Lightness method.
        METHOD_LUMINOSITY, ///< The Luminosity method.
    };

    /**
     * @brief Desaturates the image.
     *
     * @param[in,out] image
     *   The image to desaturate.
     */
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParam(const char*, const void*)
    void setParam( const char* name, const void* value );

    /**
     * @brief Sets the desaturation method to use.
     *
     * @param[in] method
     *   The method to use.
     */
    void setMethod( Method method );

protected:
    /**
     * @brief Desaturates the image using the Average method.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void desaturateAverage( IImage& dest, const IImage& src ) = 0;
    /**
     * @brief Desaturates the image using the Lightness method.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void desaturateLightness( IImage& dest, const IImage& src ) = 0;
    /**
     * @brief Desaturates the image using the Luminosity method.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void desaturateLuminosity( IImage& dest, const IImage& src ) = 0;

    /// The chosen method of desaturation.
    Method mMethod;
};

#endif /* !IDESATURATE_FILTER_HXX__INCL__ */
