/** @file
 * @brief Declaration of class IImageFilterBuilder.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#ifndef IIMAGE_FILTER_BUILDER_HXX__INCL__
#define IIMAGE_FILTER_BUILDER_HXX__INCL__

class IImageBackend;
class IImageFilter;

/**
 * @brief Interface of an image filter builder.
 *
 * @author Jan Bobek
 */
class IImageFilterBuilder
{
public:
    /**
     * @brief A virtual destructor.
     */
    virtual ~IImageFilterBuilder() {}

    /// @copydoc ImageFilterBuilder::buildFilter(ImageBackend&)
    virtual IImageFilter* buildFilter( IImageBackend& backend ) = 0;
};

#endif /* !IIMAGE_FILTER_BUILDER_HXX__INCL__ */
