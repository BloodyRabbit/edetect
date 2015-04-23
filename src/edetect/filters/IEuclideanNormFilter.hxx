/** @file
 * @brief Declaration of class IEuclideanNormFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#ifndef FILTERS__IEUCLIDEAN_NORM_FILTER_HXX__INCL__
#define FILTERS__IEUCLIDEAN_NORM_FILTER_HXX__INCL__

#include "filters/IDualInputFilter.hxx"

/**
 * @brief Interface of a filter which
 *   computes the Euclidean norm.
 *
 * @author Jan Bobek
 */
class IEuclideanNormFilter
: public IDualInputFilter
{
public:
    /// @copydoc IDualInputFilter::IDualInputFilter(IImageFilter*, IImageFilter*)
    IEuclideanNormFilter(
        IImageFilter* first = NULL,
        IImageFilter* second = NULL
        );

    /// @copydoc IDualInputFilter::filter(IImage&)
    void filter( IImage& image );
};

#endif /* !FILTERS__IEUCLIDEAN_NORM_FILTER_HXX__INCL__ */
