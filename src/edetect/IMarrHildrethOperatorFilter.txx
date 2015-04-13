/** @file
 * @brief Template definition of class IMarrHildrethOperatorFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

/*************************************************************************/
/* IMarrHildrethOperatorFilter< F1, F2 >                                 */
/*************************************************************************/
template< typename F1, typename F2 >
void
IMarrHildrethOperatorFilter< F1, F2 >::filter(
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

template< typename F1, typename F2 >
void
IMarrHildrethOperatorFilter< F1, F2 >::setParam(
    const char* name,
    const void* value
    )
{
    if( !strcmp( name, "radius1" ) )
    {
        char* endptr;
        unsigned int radius =
            strtoul( (const char*)value, &endptr, 10 );

        if( *endptr )
            throw std::invalid_argument(
                "IMarrHildrethOperatorFilter: Invalid radius1 value" );

        setRadius1( radius );
    }
    else if( !strcmp( name, "radius2" ) )
    {
        char* endptr;
        unsigned int radius =
            strtoul( (const char*)value, &endptr, 10 );

        if( *endptr )
            throw std::invalid_argument(
                "IMarrHildrethOperatorFilter: Invalid radius2 value" );

        setRadius2( radius );
    }
    else
        IImageFilter::setParam( name, value );
}

template< typename F1, typename F2 >
void
IMarrHildrethOperatorFilter< F1, F2 >::setRadius1(
    unsigned int radius
    )
{
    mLogFilt1.setRadius( radius );
}

template< typename F1, typename F2 >
void
IMarrHildrethOperatorFilter< F1, F2 >::setRadius2(
    unsigned int radius
    )
{
    mLogFilt2.setRadius( radius );
}
