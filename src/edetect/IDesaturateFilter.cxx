/** @file
 * @brief Definition of class IDesaturateFilter.
 *
 * @author Jan Bobek
 * @since 9th April 2015
 */

#include "edetect.hxx"
#include "IDesaturateFilter.hxx"
#include "IImage.hxx"

/*************************************************************************/
/* IDesaturateFilter                                                     */
/*************************************************************************/
void
IDesaturateFilter::filter(
    IImage& image
    )
{
    switch( image.format() )
    {
    case Image::FMT_GRAY_UINT8:
    case Image::FMT_GRAY_FLOAT32:
        fputs( "IDesaturateFilter: Image already in grayscale\n", stderr );
        return;

    case Image::FMT_RGB_FLOAT32:
        break;

    default:
    case Image::FMT_RGB_UINT8:
        throw std::runtime_error(
            "IDesaturateFilter: Unsupported image format" );
    }

    IImage* output = image.cloneImpl();
    output->reset( image.rows(), image.columns(),
                   Image::FMT_GRAY_FLOAT32 );

    switch( mMethod )
    {
    case METHOD_AVERAGE:
        desaturateAverage( *output, image );
        break;

    case METHOD_LIGHTNESS:
        desaturateLightness( *output, image );
        break;

    case METHOD_LUMINOSITY:
        desaturateLuminosity( *output, image );
        break;
    }

    image.swap( *output );
    delete output;
}

void
IDesaturateFilter::setParam(
    const char* name,
    const void* value
    )
{
    if( !strcmp( name, "method" ) )
    {
        const char* strval = (const char*)value;

        if( !strcmp( strval, "average" ) )
            setMethod( METHOD_AVERAGE );
        else if( !strcmp( strval, "lightness" ) )
            setMethod( METHOD_LIGHTNESS );
        else if( !strcmp( strval, "luminosity" ) )
            setMethod( METHOD_LUMINOSITY );
        else
            throw std::invalid_argument(
                "IDesaturateFilter: Method not implemented" );
    }
    else
        IImageFilter::setParam( name, value );
}

void
IDesaturateFilter::setMethod(
    IDesaturateFilter::Method method
    )
{
    mMethod = method;
}
