/** @file
 * @brief Declaration of class ImageFilterBuilder.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#ifndef IMAGE_FILTER_BUILDER_HXX__INCL__
#define IMAGE_FILTER_BUILDER_HXX__INCL__

class ImageBackend;

class IImageFilter;
class IImageFilterBuilder;

/**
 * @brief Interface of an image filter builder.
 *
 * @author Jan Bobek
 */
class ImageFilterBuilder
{
public:
    /**
     * @brief Releases the implementation.
     */
    ~ImageFilterBuilder();

    /**
     * @brief Builds the filter given a backend.
     *
     * @param[in] backend
     *   The backend to use.
     *
     * @return
     *   The built filter.
     */
    IImageFilter* buildFilter( ImageBackend& backend );

protected:
    /**
     * @brief Initializes the builder.
     *
     * @param[in] builder
     *   The builder implementation.
     */
    ImageFilterBuilder( IImageFilterBuilder* builder );

    /// Pointer to the implementation.
    IImageFilterBuilder* mBuilder;
};

/**
 * @brief A string image filter builder.
 *
 * @author Jan Bobek
 */
class StringFilterBuilder
: public ImageFilterBuilder
{
public:
    /**
     * @brief Initializes the string filter builder.
     *
     * @param[in] str
     *   The description string.
     */
    StringFilterBuilder( char* str );
};

#endif /* !IMAGE_FILTER_BUILDER_HXX__INCL__ */
