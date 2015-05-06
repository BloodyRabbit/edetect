/** @file
 * @brief Definition of class CudaTimerFilter.
 *
 * @author Jan Bobek
 * @since 1st May 2015
 */

#include "edetect.hxx"
#include "cuda/CudaError.hxx"
#include "cuda/CudaTimerFilter.hxx"

/*************************************************************************/
/* CudaTimerFilter                                                       */
/*************************************************************************/
CudaTimerFilter::CudaTimerFilter(
    IImageFilter* filter
    )
: ITimerFilter( filter )
{
    cudaCheckError( cudaEventCreate( &mStart ) );
    cudaCheckError( cudaEventCreate( &mStop ) );
}

CudaTimerFilter::~CudaTimerFilter()
{
    cudaEventDestroy( mStart );
    cudaEventDestroy( mStop );
}

void
CudaTimerFilter::filter(
    IImage& image
    )
{
    float ms;

    cudaCheckError( cudaEventRecord( mStart ) );
    mFilter->filter( image );
    cudaCheckError( cudaEventRecord( mStop ) );

    cudaCheckError( cudaEventSynchronize( mStop ) );
    cudaCheckError( cudaEventElapsedTime( &ms, mStart, mStop ) );

    print( ms, image );
}
