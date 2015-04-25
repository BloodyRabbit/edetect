/** @file
 * @brief Declaration of class IInvertFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#ifndef FILTERS__IINVERT_FILTER_HXX__INCL__
#define FILTERS__IINVERT_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of an inversion filter.
 *
 * @author Jan Bobek
 */
class IInvertFilter
: public IImageFilter
{
public:
    /**
     * @brief Applies the inversion filter to an image.
     *
     * @param[in,out] image
     *   The image to apply the operator to.
     */
    void filter( IImage& image );

protected:
    /**
     * @brief Inverts the image.
     *
     * @param[in,out] image
     *   The image to invert.
     */
    virtual void invert( IImage& image ) = 0;
};

#endif /* !FILTERS__IINVERT_FILTER_HXX__INCL__ */
