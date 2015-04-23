/** @file
 * @brief Declaration of class IZeroCrossFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#ifndef FILTERS__IZERO_CROSS_FILTER_HXX__INCL__
#define FILTERS__IZERO_CROSS_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a zero-crossing
 *   detection filter.
 *
 * @author Jan Bobek
 */
class IZeroCrossFilter
: public IImageFilter
{
public:
    /// @copydoc ImageFilter::filter(Image&)
    void filter( IImage& image );

protected:
    /**
     * @brief Detects zero-crossings in the image.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image to examine.
     */
    virtual void detectZeroCross( IImage& dest, const IImage& src ) = 0;
};

#endif /* !FILTERS__IZERO_CROSS_FILTER_HXX__INCL__ */
