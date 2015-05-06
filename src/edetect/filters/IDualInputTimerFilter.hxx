/** @file
 * @brief Declaration of class IDualInputTimerFilter.
 *
 * @author Jan Bobek
 * @since 6th May 2015
 */

#ifndef FILTERS__IDUAL_INPUT_TIMER_FILTER_HXX__INCL__
#define FILTERS__IDUAL_INPUT_TIMER_FILTER_HXX__INCL__

#include "IDualInputFilter.hxx"

/**
 * @brief Timer of a filter with two inputs.
 *
 * @author Jan Bobek
 */
class IDualInputTimerFilter
: public IDualInputFilter
{
public:
    /**
     * @brief Initializes the filter.
     *
     * @param[in,opt] filter
     *   The filter to measure.
     */
    IDualInputTimerFilter( IDualInputFilter* filter = NULL );
    /**
     * @brief Frees the dual-input filter.
     */
    ~IDualInputTimerFilter();

    /// @copydoc IDualInputFilter::setParamVa(const char*, va_list ap)
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
    void setFilter( IDualInputFilter* filter );

protected:
    /// @copydoc IDualInputFilter::filter2(IImage&, const IImage&)
    void filter2( IImage& dest, const IImage& src );

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
    /// The filter to measure.
    IDualInputFilter* mFilter;
};

#endif /* !FILTERS__IDUAL_INPUT_TIMER_FILTER_HXX__INCL__ */
