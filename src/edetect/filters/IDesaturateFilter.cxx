/** @file
 * @brief Definition of class IDesaturateFilter.
 *
 * @author Jan Bobek
 * @since 9th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "filters/IDesaturateFilter.hxx"

/*************************************************************************/
/* IDesaturateFilter                                                     */
/*************************************************************************/
void
IDesaturateFilter::filter(
    IImage& image
    )
{
    IImage* output;

    switch( image.format() )
    {
    case Image::FMT_GRAY_UINT8:
    case Image::FMT_GRAY_FLOAT32:
        fputs( "IDesaturateFilter: Image already in grayscale\n", stderr );
        return;

    case Image::FMT_RGB_UINT8:
        output = image.cloneImpl();
        output->reset( image.rows(), image.columns(),
                       Image::FMT_GRAY_UINT8 );

        (this->*mDesaturateInt)( *output, image );
        break;

    case Image::FMT_RGB_FLOAT32:
        output = image.cloneImpl();
        output->reset( image.rows(), image.columns(),
                       Image::FMT_GRAY_FLOAT32 );

        (this->*mDesaturateFloat)( *output, image );
        break;

    default:
        throw std::runtime_error(
            "IDesaturateFilter: Unsupported image format" );
    }

    image.swap( *output );
    delete output;
}

void
IDesaturateFilter::setParamVa(
    const char* name,
    va_list ap
    )
{
    const char* strval;

    if( !strcmp( name, "method" ) )
    {
        strval = va_arg( ap, const char* );

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
        IImageFilter::setParamVa( name, ap );
}

void
IDesaturateFilter::setMethod(
    IDesaturateFilter::Method method
    )
{
    switch( method )
    {
    case METHOD_AVERAGE:
        mDesaturateInt   = &IDesaturateFilter::desaturateAverageInt;
        mDesaturateFloat = &IDesaturateFilter::desaturateAverageFloat;
        break;

    case METHOD_LIGHTNESS:
        mDesaturateInt   = &IDesaturateFilter::desaturateLightnessInt;
        mDesaturateFloat = &IDesaturateFilter::desaturateLightnessFloat;
        break;

    case METHOD_LUMINOSITY:
        mDesaturateInt   = &IDesaturateFilter::desaturateLuminosityInt;
        mDesaturateFloat = &IDesaturateFilter::desaturateLuminosityFloat;
        break;
    }
}
