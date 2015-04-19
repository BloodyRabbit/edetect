/** @file
 * @brief Declaration of class IMultiplyFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#ifndef IMULTIPLY_FILTER_HXX__INCL__
#define IMULTIPLY_FILTER_HXX__INCL__

#include "IDualInputFilter.hxx"

/**
 * @brief Interface of a filter which
 *   multiplies the image with another.
 *
 * @author Jan Bobek
 */
class IMultiplyFilter
: public IDualInputFilter
{
public:
    /// @copydoc IDualInputFilter::IDualInputFilter(IImageFilter*, IImageFilter*)
    IMultiplyFilter(
        IImageFilter* first = NULL,
        IImageFilter* second = NULL
        );

    /// @copydoc IDualInputFilter::filter(IImage&)
    void filter( IImage& image );
};

#endif /* !IMULTIPLY_FILTER_HXX__INCL__ */
