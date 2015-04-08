/** @file
 * @brief Declaration of class IIntFloatFilter.
 *
 * @author Jan Bobek
 * @since 9th April 2015
 */

#ifndef IINT_FLOAT_FILTER_HXX__INCL__
#define IINT_FLOAT_FILTER_HXX__INCL__

#include "IImage.hxx"
#include "IImageFilter.hxx"

/**
 * @brief Interface of an integer-pixel/float-pixel
 *   conversion filter.
 *
 * @author Jan Bobek
 */
class IIntFloatFilter
: public IImageFilter
{
public:
    /// @copydoc ImageFilter::filter(Image&)
    void filter( IImage& image );

protected:
    /**
     * @brief Converts integer-pixels to float-pixels.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void convertInt2Float( IImage& dest, const IImage& src ) = 0;
    /**
     * @brief Converts float-pixels to integer-pixels.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void convertFloat2Int( IImage& dest, const IImage& src ) = 0;

    /// Table of target formats by format
    static const Image::Format FMT_TARGET[];
};

#endif /* !IINT_FLOAT_FILTER_HXX__INCL__ */
