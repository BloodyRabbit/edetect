/** @file
 * @brief Declaration of class INonmaxSuppressionFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#ifndef FILTERS__INONMAX_SUPPRESSION_FILTER_HXX__INCL__
#define FILTERS__INONMAX_SUPPRESSION_FILTER_HXX__INCL__

#include "filters/IDualInputFilter.hxx"

/**
 * @brief Interface of a non-maximum
 *   supression filter.
 *
 * @author Jan Bobek
 */
class INonmaxSuppressionFilter
: public IDualInputFilter
{
public:
    /// @copydoc IDualInputFilter::IDualInputFilter(IImageFilter*, IImageFilter*)
    INonmaxSuppressionFilter(
        IImageFilter* first = NULL,
        IImageFilter* second = NULL
        );

    /// @copydoc IDualInputFilter::filter(IImage&)
    void filter( IImage& image );

protected:
    /// @copydoc IDualInputFilter::filter2(IImage&, const IImage&)
    void filter2( IImage& first, const IImage& second );

    /**
     * @brief Applies non-maximum suppression to the images.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] first
     *   The first input image.
     * @param[in] second
     *   The second input image.
     */
    virtual void nonmaxSuppress(
        IImage& dest,
        const IImage& first,
        const IImage& second
        ) = 0;
};

#endif /* !FILTERS__INONMAX_SUPPRESSION_FILTER_HXX__INCL__ */
