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
     * @brief Initializes the builder.
     *
     * @param[in] str
     *   The description string.
     */
    ImageFilterBuilder( char* str );
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
    /// Pointer to the implementation.
    IImageFilterBuilder* mBuilder;
};

#endif /* !IMAGE_FILTER_BUILDER_HXX__INCL__ */
