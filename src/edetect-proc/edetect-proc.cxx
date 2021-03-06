/** @file
 * @brief This program processes images using edetect library.
 *
 * @author Jan Bobek
 */

#include "edetect-proc.hxx"

int
main(
    int argc,
    char* argv[]
    )
{
    if( argc < 3 || !(argc % 2) )
    {
        fprintf( stderr, "Usage: %s <backend> <filter.xml> [<infile> <outfile> ...]\n",
                 argv[0] );
        return EXIT_FAILURE;
    }

    try
    {
        fprintf( stderr, "Initializing backend `%s'\n", argv[1] );
        ImageBackend backend( argv[1] );

        fprintf( stderr, "Loading filter definition from `%s'\n", argv[2] );
        XmlFilterBuilder builder( argv[2] );
        ImageFilter filter( backend, builder );

        argc -= 3;
        argv += 3;
        for(; 0 < argc; argc -= 2, argv += 2 )
        {
            fprintf( stderr, "\nLoading image from file `%s'\n",
                     argv[0] );
            Image image( backend, argv[0] );

            fprintf( stderr,
                     "Input image info:\n"
                     "  Width:      %u px\n"
                     "  Height:     %u px\n"
                     "  Pixel size: %u B\n",
                     image.columns(),
                     image.rows(),
                     image.pixelSize() );

            filter.filter( image );

            fprintf( stderr,
                     "Output image info:\n"
                     "  Width:      %u px\n"
                     "  Height:     %u px\n"
                     "  Pixel size: %u B\n",
                     image.columns(),
                     image.rows(),
                     image.pixelSize() );

            image.save( argv[1] );
            fprintf( stderr, "Saved image to file `%s'\n",
                     argv[1] );
        }
    }
    catch( const std::exception& e )
    {
        fprintf( stderr, "%s\n", e.what() );
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
