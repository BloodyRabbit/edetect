/** @file
 * @brief Declaration of class IDualInputFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#ifndef IDUAL_INPUT_FILTER_HXX__INCL__
#define IDUAL_INPUT_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a filter with two inputs.
 *
 * @author Jan Bobek
 */
class IDualInputFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     *
     * @param[in,opt] first
     *   The first filter.
     * @param[in,opt] second
     *   The second filter.
     */
    IDualInputFilter(
        IImageFilter* first = NULL,
        IImageFilter* second = NULL
        );
    /**
     * @brief Frees the filters.
     */
    ~IDualInputFilter();

    /// @copydoc IImageFilter::filter(IImage&)
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParam(const char*, const void*)
    void setParam( const char* name, const void* value );

    /**
     * @brief Sets the first input filter.
     *
     * @param[in] filter
     *   The filter to use.
     */
    void setFirst( IImageFilter* filter );
    /**
     * @brief Sets the second input filter.
     *
     * @param[in] filter
     *   The filter to use.
     */
    void setSecond( IImageFilter* filter );

protected:
    /**
     * @brief Filters the two images.
     *
     * @param[in,out] first
     *   Image filtered by the first filter.
     * @param[in] second
     *   Image filtered by the second filter.
     */
    virtual void filter2( IImage& first, const IImage& second ) = 0;

    /// The first input filter.
    IImageFilter* mFirst;
    /// The second input filter.
    IImageFilter* mSecond;
};

#endif /* !IDUAL_INPUT_FILTER_HXX__INCL__ */
