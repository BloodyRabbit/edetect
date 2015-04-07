/** @file
 * @brief Definition of class ImageFilter.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#include "edetect.hxx"
#include "Image.hxx"
#include "ImageBackend.hxx"
#include "ImageFilter.hxx"

#include "IImageFilter.hxx"

/*************************************************************************/
/* ImageFilter                                                           */
/*************************************************************************/
ImageFilter::ImageFilter(
    ImageBackend& backend,
    const char* name,
    unsigned int nparams,
    ...
    )
: mFilter( backend.createFilter( name ) )
{
    va_list ap;
    va_start( ap, nparams );

    const char* pname;
    const void* pval;
    while( nparams-- )
    {
        pname = va_arg( ap, const char* );
        pval = va_arg( ap, const void* );

        setParam( pname, pval );
    }

    va_end( ap );
}

ImageFilter::~ImageFilter()
{
    delete mFilter;
}

void
ImageFilter::filter(
    Image& image
    )
{
    mFilter->filter( *image.mImage );
}

void
ImageFilter::setParam(
    const char* name,
    const void* value
    )
{
    mFilter->setParam( name, value );
}
