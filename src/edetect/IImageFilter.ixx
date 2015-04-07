/** @file
 * @brief Inline definition of class IImageFilter.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#include "IImage.hxx"

/*************************************************************************/
/* IImageFilter                                                          */
/*************************************************************************/
inline
IImageFilter::~IImageFilter()
{
}

inline
void
IImageFilter::filter(
    IImage& image
    )
{
    image.apply( *this );
}

inline
void
IImageFilter::filter(
    CudaImage&
    )
{
    throw std::invalid_argument(
        "IImageFilter: Filtering CudaImage not implemented" );
}

inline
void
IImageFilter::setParam(
    const char*,
    const void*
    )
{
    throw std::invalid_argument(
        "IImageFilter: Parameter not implemented" );
}
