/** @file
 * @brief Definition of class CpuTimerFilter.
 *
 * @author Jan Bobek
 * @since 1st May 2015
 */

#include "edetect.hxx"
#include "cpu/CpuTimerFilter.hxx"

/*************************************************************************/
/* CpuTimerFilter                                                        */
/*************************************************************************/
void
CpuTimerFilter::filter(
    IImage& image
    )
{
    float ms;
    timeval start, end;

    if( gettimeofday( &start, NULL ) )
        throw std::runtime_error(
            "CpuTimerFilter: starting call to `gettimeofday' failed" );

    ITimerFilter::filter( image );

    if( gettimeofday( &end, NULL ) )
        throw std::runtime_error(
            "CpuTimerFilter: ending call to `gettimeofday' failed" );

    timersub( &end, &start, &end );

    ms = end.tv_sec * 1e+3 +
        end.tv_usec * 1e-3;
    print( ms, image );
}
