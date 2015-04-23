/** @file
 * @brief Declaration of class IKirschOperatorFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef FILTERS__IKIRSCH_OPERATOR_FILTER_HXX__INCL__
#define FILTERS__IKIRSCH_OPERATOR_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a Kirsch operator filter.
 *
 * @author Jan Bobek
 */
class IKirschOperatorFilter
: public IImageFilter
{
public:
    /**
     * @brief Applies the Kirsch operator to an image.
     *
     * @param[in,out] image
     *   The image to apply the operator to.
     */
    void filter( IImage& image );

protected:
    /**
     * @brief Applies the Kirsch operator to the image.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The source image data.
     */
    virtual void applyKirschOperator( IImage& dest, const IImage& src ) = 0;
};

#endif /* !FILTERS__IKIRSCH_OPERATOR_FILTER_HXX__INCL__ */
