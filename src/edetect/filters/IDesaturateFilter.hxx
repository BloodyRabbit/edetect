/** @file
 * @brief Declaration of class IDesaturateFilter.
 *
 * @author Jan Bobek
 * @since 9th April 2015
 */

#ifndef FILTERS__IDESATURATE_FILTER_HXX__INCL__
#define FILTERS__IDESATURATE_FILTER_HXX__INCL__

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
    /// @copydoc IImageFilter::setParamVa(const char*, va_list ap)
    void setParamVa( const char* name, va_list ap );

    /**
     * @brief Sets the desaturation method to use.
     *
     * @param[in] method
     *   The method to use.
     */
    void setMethod( Method method );

protected:
    /**
     * @brief Desaturates an integer image using the Average method.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void desaturateAverageInt( IImage& dest, const IImage& src ) = 0;
    /**
     * @brief Desaturates a floating-point image using the Average method.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void desaturateAverageFloat( IImage& dest, const IImage& src ) = 0;

    /**
     * @brief Desaturates an integer image using the Lightness method.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void desaturateLightnessInt( IImage& dest, const IImage& src ) = 0;
    /**
     * @brief Desaturates a floating-point image using the Lightness method.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void desaturateLightnessFloat( IImage& dest, const IImage& src ) = 0;

    /**
     * @brief Desaturates an integer image using the Luminosity method.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void desaturateLuminosityInt( IImage& dest, const IImage& src ) = 0;
    /**
     * @brief Desaturates a floating-point image using the Luminosity method.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void desaturateLuminosityFloat( IImage& dest, const IImage& src ) = 0;

    /// The chosen integer desaturation callback
    void (IDesaturateFilter::* mDesaturateInt)(IImage&, const IImage&);
    /// The chosen floating-point desaturation callback
    void (IDesaturateFilter::* mDesaturateFloat)(IImage&, const IImage&);
};

#endif /* !FILTERS__IDESATURATE_FILTER_HXX__INCL__ */
