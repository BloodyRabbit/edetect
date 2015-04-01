/** @file
 * @brief This program applies Gaussian blur to a given image.
 *
 * @author Jan Bobek
 */

#include "edetect-proc.hxx"
#include "ImageFilterPipeline.hxx"

int
main(
    int argc,
    char* argv[]
    )
{
    const char *radius, *infile, *outfile;

    if( argc != 4 )
    {
        fprintf( stderr, "Usage: %s <gauss-radius> <infile> <outfile>\n",
                 argv[0] );
        return false;
    }

    radius = argv[1];
    infile = argv[2];
    outfile = argv[3];

    try
    {
        ImageBackend backend( "cuda" );
        Image image( backend, infile );

        fprintf( stderr,
                 "Loaded image from file `%s'\n"
                 "Input image info:\n"
                 "  Width:     %u px\n"
                 "  Height:    %u px\n",
                 infile,
                 image.columns(),
                 image.rows() );

        ImageFilterPipeline pipeline;
        // Convert image from integer RGB to float RGB
        pipeline.add( new ImageFilter( backend, "int-float" ) );
        // Convert image to float grayscale, Luminosity method
        pipeline.add( new ImageFilter( backend, "desaturate", 1,
                                       "method", "luminosity" ) );
        // Apply Gaussian blur to the image
        pipeline.add( new ImageFilter( backend, "gaussian-blur", 1,
                                       "radius", radius ) );
        // Convert the image back to integer grayscale
        pipeline.add( new ImageFilter( backend, "int-float" ) );

        pipeline.filter( image );
        image.save( outfile );

        fprintf( stderr, "Saved image to file `%s'\n", outfile );
    }
    catch( const std::exception& e )
    {
        fprintf( stderr, "%s\n", e.what() );
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
