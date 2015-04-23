/** @file
 * @brief Declaration of class XmlFilterBuilderImpl.
 *
 * @author Jan Bobek
 * @since 23th April 2015
 */

#ifndef XML_FILTER_BUILDER_IMPL_HXX__INCL__
#define XML_FILTER_BUILDER_IMPL_HXX__INCL__

#include "IImageFilterBuilder.hxx"

/**
 * @brief A XML-based filter builder.
 *
 * @author Jan Bobek
 */
class XmlFilterBuilderImpl
: public IImageFilterBuilder
{
public:
    /**
     * @brief Initializes the filter builder.
     *
     * @param[in] filename
     *   Name of the configuration file.
     */
    XmlFilterBuilderImpl( const char* filename );

    /// @copydoc IImageFilterBuilder::buildFilter(IImageBackend&)
    IImageFilter* buildFilter( IImageBackend& backend );

protected:
    /**
     * @brief Produces a filter by parsing
     *   an XML element.
     *
     * @param[in] backend
     *   The backend to use.
     * @param[in] element
     *   The element to parse.
     *
     * @return
     *   The built filter.
     */
    IImageFilter* parseFilter(
        IImageBackend& backend,
        const tinyxml2::XMLElement& element
        );
    /**
     * @brief Parses and sets a filter parameter.
     *
     * @param[in] backend
     *   The backend to use.
     * @param[out] filter
     *   The filter to parametrize.
     * @param[in] element
     *   The element to parse.
     */
    void parseParam(
        IImageBackend& backend,
        IImageFilter& filter,
        const tinyxml2::XMLElement& element
        );
    /**
     * @brief Parses and sets a filter parameter.
     *
     * @param[in] backend
     *   The backend to use.
     * @param[out] filter
     *   The filter to parametrize.
     * @param[in] name
     *   Name of the parameter to set.
     * @param[in] element
     *   The element to parse.
     */
    void parseParamElem(
        IImageBackend& backend,
        IImageFilter& filter,
        const char* name,
        const tinyxml2::XMLElement& element
        );
    /**
     * @brief Parses and sets a filter parameter.
     *
     * @param[out] filter
     *   The filter to parametrize.
     * @param[in] name
     *   Name of the parameter to set.
     * @param[in] txt
     *   The text node to parse.
     */
    void parseParamText(
        IImageFilter& filter,
        const char* name,
        const tinyxml2::XMLText& txt
        );
    /**
     * @brief Produces an array by parsing
     *   an XML element.
     *
     * @param[out] length
     *   Length of the array.
     * @param[in] element
     *   The element to parse.
     *
     * @return
     *   The parsed array.
     */
    float* parseArray(
        unsigned int& length,
        const tinyxml2::XMLElement& element
        );
    /**
     * @brief Fills an array by parsing
     *   an XML text node.
     *
     * @param[in] arrval
     *   The current array.
     * @param[in,out] length
     *   Length of the array.
     * @param[in,out] capacity
     *   Allocated capacity of the array.
     * @param[in] txt
     *   The text node to parse.
     */
    void parseArrayText(
        float*& arrval,
        unsigned int& length,
        unsigned int& capacity,
        const tinyxml2::XMLText& txt
        );
    /**
     * @brief Produces a pipeline by parsing
     *   an XML element.
     *
     * @param[in] backend
     *   The backend to use.
     * @param[in] element
     *   The element to parse.
     *
     * @return
     *   The built filter.
     */
    ImageFilterPipeline* parsePipeline(
        IImageBackend& backend,
        const tinyxml2::XMLElement& element
        );

    /// The parsed XML document.
    tinyxml2::XMLDocument mDocument;
};

#endif /* !XML_FILTER_BUILDER_IMPL_HXX__INCL__ */
