/** @file
 * @brief Declaration of ImageFilterPipeline class.
 *
 * @author Jan Bobek
 */

#ifndef IMAGE_FILTER_PIPELINE_HXX__INCL__
#define IMAGE_FILTER_PIPELINE_HXX__INCL__

/**
 * @brief A series of filters.
 *
 * @author Jan Bobek
 */
class ImageFilterPipeline
{
public:
    /**
     * @brief Releases the filters.
     */
    ~ImageFilterPipeline();

    /**
     * @brief Adds a filter to the pipeline.
     *
     * @param[in] filter
     *   The filter to add.
     */
    void add( ImageFilter* filter );
    /**
     * @brief Clears filters in the pipeline.
     */
    void clear();

    /**
     * @brief Applies the filters to an image.
     *
     * @param[in,out] image
     *   The image to filter.
     */
    void filter( Image& image );

protected:
    /// The list of active filters.
    std::list< ImageFilter* > mFilters;
};

#endif /* !IMAGE_FILTER_PIPELINE_HXX__INCL__ */
