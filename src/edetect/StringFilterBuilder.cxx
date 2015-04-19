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
#include "StringFilterBuilder.hxx"

/*************************************************************************/
/* StringFilterBuilder                                                   */
/*************************************************************************/
IImageFilter*
StringFilterBuilder::parseFilter(
    IImageBackend& backend,
    unsigned int& idx
    )
{
    char x, *name;
    unsigned int nlen;
    IImageFilter* filter;

    name = &mStr[idx];
    nlen = strcspn( name, ",:=()" );
    if( '=' == name[nlen] )
        throw std::invalid_argument(
            "Malformed filter name: unexpected `='" );
    else if( '(' == name[nlen] )
    {
        if( 0 < nlen )
            throw std::invalid_argument(
                "Malformed filter name: unexpected `('" );

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
StringFilterBuilder::parseParams(
    IImageBackend& backend,
    IImageFilter& filter,
    unsigned int& idx
    )
{
    char x, *name, *value;
    unsigned int nlen, vlen;

    while( ':' == mStr[idx] )
    {
        name = &mStr[++idx];
        nlen = strcspn( name, ",:=()" );
        if( '=' != name[nlen] )
            throw std::invalid_argument(
                "Malformed filter parameter: missing value" );

        name[nlen] = '\0';
        idx += nlen + 1;

        value = &mStr[idx];
        vlen = strcspn( value, ",:=()" );
        if( '=' == value[vlen] )
            throw std::invalid_argument(
                "Malformed filter parameter: unexpected `=' in value" );
        else if( '(' == value[vlen] )
        {
            if( 0 < vlen )
                throw std::invalid_argument(
                    "Malformed filter parameter: unexpected '(' in value" );

            filter.setParam(
                name, parsePipeline( backend, ++idx ) );
        }
        else
        {
            x = value[vlen], value[vlen] = '\0';
            filter.setParam( name, value );
            value[vlen] = x;

            idx += vlen;
        }

        name[nlen] = '=';
    }
}

ImageFilterPipeline*
StringFilterBuilder::parsePipeline(
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

    if( ')' != mStr[idx - 1] )
        throw std::invalid_argument(
            "Malformed pipeline: missing `)'" );

    return pipeline;
}

StringFilterBuilder::StringFilterBuilder(
    char* str
    )
: mStr( str )
{
}

IImageFilter*
StringFilterBuilder::buildFilter(
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