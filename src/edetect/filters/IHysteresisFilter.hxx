/** @file
 * @brief Declaration of class IHysteresisFilter.
 *
 * @author Jan Bobek
 * @since 26th April 2015
 */

#ifndef FILTERS__IHYSTERESIS_FILTER_HXX__INCL__
#define FILTERS__IHYSTERESIS_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a hysteresis filter.
 *
 * @author Jan Bobek
 */
class IHysteresisFilter
: public IImageFilter
{
public:
    /// @copydoc IImageFilter::filter(IImage&)
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParamVa(const char*, va_list ap)
    void setParamVa( const char* name, va_list ap );

    /**
     * @brief Sets the low threshold.
     *
     * @param[in] threshold
     *   The threshold value to set.
     */
    void setThresholdLow( float threshold );
    /**
     * @brief Sets the high threshold.
     *
     * @param[in] threshold
     *   The threshold value to set.
     */
    void setThresholdHigh( float threshold );

protected:
    /**
     * @brief Applies hysteresis to the image.
     *
     * @param[out] dest
     *   Where to place the result.
     * @param[in] src
     *   The source image data.
     */
    virtual void hysteresis( IImage& dest, const IImage& src ) = 0;

    /// The low threshold.
    float mThresholdLow;
    /// The high threshold.
    float mThresholdHigh;
};

#endif /* !FILTERS__IHYSTERESIS_FILTER_HXX__INCL__ */
