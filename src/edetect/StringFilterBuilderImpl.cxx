/** @file
 * @brief Definition of class ImageFilterBuilder.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#include "edetect.hxx"
#include "IImageBackend.hxx"
#include "IImageFilter.hxx"
#include "ImageFilterPipeline.hxx"
#include "StringFilterBuilderImpl.hxx"

/*************************************************************************/
/* StringFilterBuilderImpl                                               */
/*************************************************************************/
IImageFilter*
StringFilterBuilderImpl::parseFilter(
    IImageBackend& backend,
    unsigned int& idx
    )
{
    char x, *name;
    unsigned int nlen;
    IImageFilter* filter;

    name = &mStr[idx];
    nlen = strcspn( name, ",:=[]{}" );
    if( '=' == name[nlen] )
        throw std::invalid_argument(
            "Malformed filter name: unexpected `='" );
    else if( '[' == name[nlen] )
        throw std::invalid_argument(
            "Malformed filter name: unexpeced `['" );
    else if( ']' == name[nlen] )
        throw std::invalid_argument(
            "Malformed filter name: unexpected `]'" );
    else if( '{' == name[nlen] )
    {
        if( 0 < nlen )
            throw std::invalid_argument(
                "Malformed filter name: unexpected `{'" );

        filter = parsePipeline( backend, ++idx );
    }
    else
    {
        x = name[nlen], name[nlen] = '\0';
        filter = backend.createFilter( name );
        name[nlen] = x;

        parseParams( backend, *filter, idx += nlen );
    }

    return filter;
}

void
StringFilterBuilderImpl::parseParams(
    IImageBackend& backend,
    IImageFilter& filter,
    unsigned int& idx
    )
{
    float* arrval;
    char x, *name, *strval, *endp;
    unsigned int nlen, vlen, intval;

    while( ':' == mStr[idx] )
    {
        name = &mStr[++idx];
        nlen = strcspn( name, ",:=[]{}" );
        if( '=' != name[nlen] )
            throw std::invalid_argument(
                "Malformed filter parameter: missing value" );

        name[nlen] = '\0';
        idx += nlen + 1;

        strval = &mStr[idx];
        vlen = strcspn( strval, ",:=[]{}" );
        if( '=' == strval[vlen] )
            throw std::invalid_argument(
                "Malformed filter parameter: unexpected `=' in value" );
        else if( '[' == strval[vlen] )
        {
            if( 0 < vlen )
                throw std::invalid_argument(
                    "Malformed filter parameter: unexpected '[' in value" );

            arrval = parseList( intval, ++idx );
            filter.setParam( name, arrval, intval );
        }
        else if( ']' == strval[vlen] )
            throw std::invalid_argument(
                "Malformed filter parameter: unexpected `]' in value" );
        else if( '{' == strval[vlen] )
        {
            if( 0 < vlen )
                throw std::invalid_argument(
                    "Malformed filter parameter: unexpected '{' in value" );

            filter.setParam(
                name, parsePipeline( backend, ++idx ) );
        }
        else
        {
            x = strval[vlen], strval[vlen] = '\0';

            /* try to convert to integer */
            intval = strtoul( strval, &endp, 10 );
            if( !*endp )
                filter.setParam( name, intval );
            else
                filter.setParam( name, strval );

            strval[vlen] = x;
            idx += vlen;
        }

        name[nlen] = '=';
    }
}

float*
StringFilterBuilderImpl::parseList(
    unsigned int& length,
    unsigned int& idx
    )
{
    char* endp;
    float* arrval;
    unsigned int capacity;

    length = 0, capacity = 4;
    arrval = (float*)malloc( capacity * sizeof(*arrval) );

    do
    {
        if( capacity <= length )
            arrval = (float*)realloc( arrval, (capacity *= 2) * sizeof(*arrval) );

        arrval[length] = strtod( &mStr[idx], &endp );
        if( ']' != *endp && !isspace( *endp ) )
            throw std::invalid_argument(
                "Malformed array: Invalid value encountered" );

        ++length;
        idx += endp - &mStr[idx] + 1;
    } while( ']' != *endp );

    arrval = (float*)realloc( arrval, length * sizeof(*arrval) );
    return arrval;
}

ImageFilterPipeline*
StringFilterBuilderImpl::parsePipeline(
    IImageBackend& backend,
    unsigned int& idx
    )
{
    ImageFilterPipeline* pipeline =
        new ImageFilterPipeline;

    do
    {
        pipeline->add(
            parseFilter( backend, idx ) );
    } while( ',' == mStr[idx++] );

    if( '}' != mStr[idx - 1] )
        throw std::invalid_argument(
            "Malformed pipeline: missing `}'" );

    return pipeline;
}

StringFilterBuilderImpl::StringFilterBuilderImpl(
    char* str
    )
: mStr( str )
{
}

IImageFilter*
StringFilterBuilderImpl::buildFilter(
    IImageBackend& backend
    )
{
    unsigned int idx;
    IImageFilter* filter;

    idx = 0;
    filter = parseFilter( backend, idx );
    if( '\0' != mStr[idx] )
        throw std::runtime_error(
            "Malformed description string: parsing incomplete" );

    return filter;
}
