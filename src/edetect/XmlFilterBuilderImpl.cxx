/** @file
 * @brief Definition of class XmlFilterBuilderImpl.
 *
 * @author Jan Bobek
 * @since 23th April 2015
 */

#include "edetect.hxx"
#include "IImageBackend.hxx"
#include "IImageFilter.hxx"
#include "ImageFilterPipeline.hxx"
#include "XmlFilterBuilderImpl.hxx"

/*************************************************************************/
/* XmlFilterBuilderImpl                                                  */
/*************************************************************************/
XmlFilterBuilderImpl::XmlFilterBuilderImpl(
    const char* filename
    )
{
    if( tinyxml2::XML_NO_ERROR != mDocument.LoadFile( filename ) )
        throw std::runtime_error( "XML parsing failed" );
}

IImageFilter*
XmlFilterBuilderImpl::buildFilter(
    IImageBackend& backend
    )
{
    if( !mDocument.RootElement() )
        throw std::runtime_error(
            "Malformed XML document: missing root element" );

    return parseFilter( backend, *mDocument.RootElement() );
}

IImageFilter*
XmlFilterBuilderImpl::parseFilter(
    IImageBackend& backend,
    const tinyxml2::XMLElement& element
    )
{
    const char* name;
    IImageFilter* filter;

    const tinyxml2::XMLElement* param;

    if( !strcmp( element.Name(), "filter" ) )
    {
        if( !(name = element.Attribute( "name" )) )
            throw std::runtime_error(
                "Malformed filter element: missing `name' attribute" );

        filter = backend.createFilter( name );
        for( param = element.FirstChildElement(); param; param = param->NextSiblingElement() )
            parseParam( backend, *filter, *param );
    }
    else if( !strcmp( element.Name(), "pipeline" ) )
        filter = parsePipeline( backend, element );
    else
        throw std::runtime_error(
            "Malformed XML document: unknown element encountered" );

    return filter;
}

void
XmlFilterBuilderImpl::parseParam(
    IImageBackend& backend,
    IImageFilter& filter,
    const tinyxml2::XMLElement& element
    )
{
    const char* name;

    const tinyxml2::XMLNode* child;
    const tinyxml2::XMLElement* elemval;
    const tinyxml2::XMLText* txtval;

    if( strcmp( element.Name(), "param" ) )
        throw std::runtime_error(
            "Malformed XML document: expected param element" );
    if( !(name = element.Attribute( "name" )) )
        throw std::runtime_error(
            "Malformed param element: missing `name' attribute" );

    for( child = element.FirstChild(); child; child = child->NextSibling() )
        if( (elemval = child->ToElement()) )
        {
            parseParamElem( backend, filter, name, *elemval );
            break;
        }
        else if( (txtval = child->ToText()) )
        {
            parseParamText( filter, name, *txtval );
            break;
        }
        else if( !child->ToComment() )
            throw std::runtime_error(
                "Malformed param element: unexpected node encountered" );
}

void
XmlFilterBuilderImpl::parseParamElem(
    IImageBackend& backend,
    IImageFilter& filter,
    const char* name,
    const tinyxml2::XMLElement& element
    )
{
    float* arrval;
    unsigned int length;

    if( !strcmp( element.Name(), "array" ) )
    {
        arrval = parseArray( length, element );
        filter.setParam( name, arrval, length );
    }
    else
        filter.setParam( name, parseFilter( backend, element ) );
}

void
XmlFilterBuilderImpl::parseParamText(
    IImageFilter& filter,
    const char* name,
    const tinyxml2::XMLText& txt
    )
{
    unsigned int intval;
    const char* strval, *endp;

    strval = txt.Value();
    intval = strtoul( strval, (char**)&endp, 10 );

    if( !*endp )
        filter.setParam( name, intval );
    else
        filter.setParam( name, strval );
}

float*
XmlFilterBuilderImpl::parseArray(
    unsigned int& length,
    const tinyxml2::XMLElement& element
    )
{
    float* arrval;
    unsigned int capacity;

    const tinyxml2::XMLNode* child;
    const tinyxml2::XMLText* txt;

    length = 0, capacity = 4;
    arrval = (float*)malloc( capacity * sizeof(*arrval) );

    for( child = element.FirstChild(); child; child = child->NextSibling() )
        if( (txt = child->ToText()) )
            parseArrayText( arrval, length, capacity, *txt );
        else if( !child->ToComment() )
            throw std::runtime_error(
                "Malformed array element: unexpected node" );

    arrval = (float*)realloc( arrval, length * sizeof(*arrval) );
    return arrval;
}

void
XmlFilterBuilderImpl::parseArrayText(
    float*& arrval,
    unsigned int& length,
    unsigned int& capacity,
    const tinyxml2::XMLText& txt
    )
{
    const char* endp = txt.Value();

    do
    {
        if( capacity <= length )
            arrval = (float*)realloc( arrval, (capacity *= 2) * sizeof(*arrval) );

        arrval[length] = strtod( endp, (char**)&endp );
        if( *endp && !isspace( *endp ) )
            throw std::invalid_argument(
                "Malformed array element: Invalid value encountered" );

        ++length;
    } while( *endp );
}

ImageFilterPipeline*
XmlFilterBuilderImpl::parsePipeline(
    IImageBackend& backend,
    const tinyxml2::XMLElement& element
    )
{
    ImageFilterPipeline* pipeline;
    const tinyxml2::XMLElement* child;

    pipeline = new ImageFilterPipeline;
    for( child = element.FirstChildElement(); child; child = child->NextSiblingElement() )
        pipeline->add( parseFilter( backend, *child ) );

    return pipeline;
}
