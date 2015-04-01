/** @file
 * @brief This program applies Sobel operator to a given image.
 *
 * @author Jan Bobek
 */

#include "common.hxx"
#include "CudaDesaturateFilter.hxx"
#include "CudaError.hxx"
#include "CudaFilterPipeline.hxx"
#include "CudaGaussianBlurFilter.hxx"
#include "CudaImage.hxx"
#include "CudaIntFloatFilter.hxx"
#include "CudaSobelOperatorFilter.hxx"

int
main(
    int argc,
    char* argv[]
    )
{
    unsigned int radius;
    char* endptr;

    if( argc != 4 )
    {
        fprintf( stderr, "Usage: %s <gauss-radius> <infile> <outfile>\n",
                 argv[0] );
        return false;
    }

    radius = strtol( argv[1], &endptr, 10 );
    if( *endptr )
    {
        fprintf( stderr, "Failed to convert %s to integer: invalid character %c\n",
                 argv[1], *endptr );
        return false;
    }

    try
    {
        CudaImage img( argv[2] );
        fprintf( stderr, "Loaded image from file `%s'\n",
                 argv[2] );

        fprintf( stderr,
                 "Input image info:\n"
                 "  Width:     %lu px\n"
                 "  Height:    %lu px\n",
                 img.columns(),
                 img.rows() );

        CudaFilterPipeline pipeline;

        // Convert image from integer RGB to float RGB
        pipeline.add( new CudaIntFloatFilter );
        // Convert image to float grayscale, Luminosity method
        pipeline.add( new CudaDesaturateFilter(
                          CudaDesaturateFilter::METHOD_LUMINOSITY ) );
        // Apply Gaussian blur to the image
        pipeline.add( new CudaGaussianBlurFilter( radius ) );
        // Apply Sobel operator to the image
        pipeline.add( new CudaSobelOperatorFilter );
        // Convert the image back to integer grayscale
        pipeline.add( new CudaIntFloatFilter );

        pipeline.process( img );
        img.save( argv[3] );

        fprintf( stderr, "Saved image to file `%s'\n",
                 argv[3] );
    }
    catch( const std::exception& e )
    {
        fprintf( stderr, "%s\n", e.what() );

        cudaCheckError( cudaDeviceReset() );
        return EXIT_FAILURE;
    }

    cudaCheckError( cudaDeviceReset() );
    return EXIT_SUCCESS;
}
