/** @file
 * @brief Declaration of class ImageFilter.
 *
 * @author Jan Bobek
 * @since 1st April 2015
 */

#ifndef IMAGE_FILTER_HXX__INCL__
#define IMAGE_FILTER_HXX__INCL__

class Image;
class ImageBackend;

class IImageFilter;

/**
 * @brief A filter for processing images.
 *
 * @author Jan Bobek
 */
class ImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     *
     * @param[in] backend
     *   Backend of the filter.
     * @param[in] name
     *   Name of the filter.
     * @param[in,opt] nparams
     *   Number of parameters to set up.
     * @param[in,opt] ...
     *   Pairs of parameter names and parameter values
     *   to set up the parameters.
     */
    ImageFilter(
        ImageBackend& backend, const char* name,
        unsigned int nparams = 0, ...
        );
    /**
     * @brief Destroys the filter.
     */
    ~ImageFilter();

    /**
     * @brief Applies the filter to given image.
     *
     * @param[in,out] image
     *   The image to filter.
     */
    void filter( Image& image );
    /**
     * @brief Sets a parameter to a given value.
     *
     * @param[in] name
     *   Name of the parameter.
     * @param[in] value
     *   Value of the parameter.
     */
    void setParam( const char* name, const void* value );

protected:
    /// Implementation of the filter.
    IImageFilter* mFilter;
};

#endif /* !IMAGE_FILTER_HXX__INCL__ */
