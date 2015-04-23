/** @file
 * @brief Template definition of class IMarrHildrethOperatorFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

/*************************************************************************/
/* IMarrHildrethOperatorFilter< CF, ZCF >                                */
/*************************************************************************/
template< typename CF, typename ZCF >
void
IMarrHildrethOperatorFilter< CF, ZCF >::filter(
    IImage& image
    )
{
    switch( image.format() )
    {
    case Image::FMT_GRAY_FLOAT32:
        break;

    default:
    case Image::FMT_GRAY_UINT8:
    case Image::FMT_RGB_FLOAT32:
    case Image::FMT_RGB_UINT8:
        throw std::runtime_error(
            "IMarrHildrethOperatorFilter: Unsupported image format" );
    }

    IImage* dup = image.clone();
    mLogFilt1.filter( image );
    mLogFilt2.filter( *dup );

    mZeroCrossFilt.filter( image );
    mZeroCrossFilt.filter( *dup );

    mergeEdges( image, *dup );
    delete dup;
}

template< typename CF, typename ZCF >
void
IMarrHildrethOperatorFilter< CF, ZCF >::setParamVa(
    const char* name,
    va_list ap
    )
{
    unsigned int radius;

    if( !strcmp( name, "radius1" ) )
    {
        radius = va_arg( ap, unsigned int );
        setRadius1( radius );
    }
    else if( !strcmp( name, "radius2" ) )
    {
        radius = va_arg( ap, unsigned int );
        setRadius2( radius );
    }
    else
        IImageFilter::setParamVa( name, ap );
}

template< typename CF, typename ZCF >
void
IMarrHildrethOperatorFilter< CF, ZCF >::setRadius1(
    unsigned int radius
    )
{
    mLogFilt1.setRadius( radius );
}

template< typename CF, typename ZCF >
void
IMarrHildrethOperatorFilter< CF, ZCF >::setRadius2(
    unsigned int radius
    )
{
    mLogFilt2.setRadius( radius );
}
