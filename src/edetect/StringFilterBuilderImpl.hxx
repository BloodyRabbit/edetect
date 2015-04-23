/** @file
 * @brief Declaration of class StringFilterBuilderImpl.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#ifndef STRING_FILTER_BUILDER_IMPL_HXX__INCL__
#define STRING_FILTER_BUILDER_IMPL_HXX__INCL__

#include "IImageFilterBuilder.hxx"

class ImageFilterPipeline;

/**
 * @brief A string filter builder.
 *
 * @author Jan Bobek
 */
class StringFilterBuilderImpl
: public IImageFilterBuilder
{
public:
    /**
     * @brief Initializes the string filter builder.
     *
     * @param[in] str
     *   The description string.
     */
    StringFilterBuilderImpl( char* str );

    /// @copydoc IImageFilterBuilder::buildFilter(IImageBackend&)
    IImageFilter* buildFilter( IImageBackend& backend );

protected:
    /**
     * @brief Produces a filter by parsing
     *   the description string.
     *
     * @param[in] backend
     *   The backend to use.
     * @param[in,out] idx
     *   Index in the description string.
     *
     * @return
     *   The built filter.
     */
    IImageFilter* parseFilter(
        IImageBackend& backend,
        unsigned int& idx
        );
    /**
     * @brief Parses and sets filter parameters.
     *
     * @param[in] backend
     *   The backend to use.
     * @param[out] filter
     *   The filter to parametrize.
     * @param[in,out] idx
     *   Index in the description string.
     */
    void parseParams(
        IImageBackend& backend,
        IImageFilter& filter,
        unsigned int& idx
        );
    /**
     * @brief Produces a pipeline by parsing
     *   the description string.
     *
     * @param[in] backend
     *   The backend to use.
     * @param[in,out] idx
     *   Index in the description string.
     *
     * @return
     *   The built filter.
     */
    ImageFilterPipeline* parsePipeline(
        IImageBackend& backend,
        unsigned int& idx
        );

    /// The description string.
    char* mStr;
};

#endif /* !STRING_FILTER_BUILDER_IMPL_HXX__INCL__ */
