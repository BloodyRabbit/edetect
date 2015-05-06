/** @file
 * @brief Declaration of class ITimerFilter.
 *
 * @author Jan Bobek
 * @since 1st May 2015
 */

#ifndef FILTERS__ITIMER_FILTER_HXX__INCL__
#define FILTERS__ITIMER_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a timer filter.
 *
 * @author Jan Bobek
 */
class ITimerFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the timer filter.
     *
     * @param[in] filter
     *   The filter to measure.
     */
    ITimerFilter( IImageFilter* filter = NULL );
    /**
     * @brief Releases the filter.
     */
    ~ITimerFilter();

    /// @copydoc IImageFilter::setParamVa(const char*, va_list ap)
    void setParamVa( const char* name, va_list ap );

    /**
     * @brief Sets the timer name.
     *
     * @param[in] name
     *   Name of the timer.
     */
    void setName( const char* name );
    /**
     * @brief Sets the filter to measure.
     *
     * @param[in] filter
     *   The filter to measure.
     */
    void setFilter( IImageFilter* filter );

protected:
    /**
     * @brief Prints the measured time interval.
     *
     * @param[in] ms
     *   The interval in milliseconds.
     * @param[in] image
     *   The processed image.
     */
    void print( float ms, const IImage& image );

    /// Name of this timer.
    char* mName;
    /// The filter being measured.
    IImageFilter* mFilter;
};

#endif /* !FILTERS__ITIMER_FILTER_HXX__INCL__ */
