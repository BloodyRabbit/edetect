/** @file
 * @brief Definition of class CudaDualInputTimerFilter.
 *
 * @author Jan Bobek
 * @since 6th May 2015
 */

#include "edetect.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaDualInputTimerFilter.hxx"

/*************************************************************************/
/* CudaDualInputTimerFilter                                              */
/*************************************************************************/
CudaDualInputTimerFilter::CudaDualInputTimerFilter(
    IDualInputFilter* filter
    )
: IDualInputTimerFilter( filter )
{
    cudaCheckError( cudaEventCreate( &mStart ) );
    cudaCheckError( cudaEventCreate( &mStop ) );
}

CudaDualInputTimerFilter::~CudaDualInputTimerFilter()
{
    cudaEventDestroy( mStart );
    cudaEventDestroy( mStop );
}

void
CudaDualInputTimerFilter::filter2(
    IImage& dest,
    const IImage& src
    )
{
    float ms;

    cudaCheckError( cudaEventRecord( mStart ) );
    IDualInputTimerFilter::filter2( dest, src );
    cudaCheckError( cudaEventRecord( mStop ) );

    cudaCheckError( cudaEventSynchronize( mStop ) );
    cudaCheckError( cudaEventElapsedTime( &ms, mStart, mStop ) );

    print( ms, dest );
}
