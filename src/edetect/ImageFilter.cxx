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
#include "ImageFilterBuilder.hxx"

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
    while( nparams-- )
    {
        pname = va_arg( ap, const char* );
        setParamVa( pname, ap );
    }

    va_end( ap );
}

ImageFilter::ImageFilter(
    ImageBackend& backend,
    ImageFilterBuilder& builder
    )
: mFilter( builder.buildFilter( backend ) )
{
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
    ...
    )
{
    va_list ap;
    va_start( ap, name );
    setParamVa( name, ap );
    va_end( ap );
}

void
ImageFilter::setParamVa(
    const char* name,
    va_list ap
    )
{
    mFilter->setParamVa( name, ap );
}
