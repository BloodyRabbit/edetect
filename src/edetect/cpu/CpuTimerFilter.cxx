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
    timeval start, end;

    if( gettimeofday( &start, NULL ) )
        throw std::runtime_error(
            "CpuTimerFilter: starting call to `gettimeofday' failed" );

    mFilter->filter( image );

    if( gettimeofday( &end, NULL ) )
        throw std::runtime_error(
            "CpuTimerFilter: ending call to `gettimeofday' failed" );

    timersub( &end, &start, &end );
    fprintf( stdout, "Timer `%s': %ld.%ld milliseconds\n",
             mName, end.tv_sec * 1000 + end.tv_usec / 1000,
             end.tv_usec % 1000 );
}
