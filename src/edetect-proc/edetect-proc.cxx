/** @file
 * @brief This program processes images using edetect library.
 *
 * @author Jan Bobek
 */

#include "edetect-proc.hxx"
#include "ImageFilterPipeline.hxx"

void
buildPipeline(
    char* pipestr,
    ImageBackend& backend,
    ImageFilterPipeline& pipeline
    )
{
    bool param, cont;
    size_t a, b;
    ImageFilter* filt;

    fprintf( stderr, "Building pipeline `%s':\n",
             pipestr );
    cont = true;

    while( cont )
    {
        a = strcspn( pipestr, ",:=" );
        if( '=' == pipestr[a] )
            throw std::invalid_argument(
                "Malformed filter name: unexpected `='" );

        param = (':' == pipestr[a]);
        cont = ('\0' != pipestr[a]);
        pipestr[a] = '\0';

        fprintf( stderr, "  Creating filter `%s'\n",
                 pipestr );
        filt = new ImageFilter( backend, pipestr );
        pipestr += a + 1;

        while( param )
        {
            a = strcspn( pipestr, ",:=" );
            if( '=' != pipestr[a] )
                throw std::invalid_argument(
                    "Malformed filter parameter: missing value" );

            b = strcspn( &pipestr[a + 1], ",:=" );
            if( '=' == pipestr[a + b + 1] )
                throw std::invalid_argument(
                    "Malformed filter parameter: unexpected `=' in value" );

            param = (':' == pipestr[a + b + 1]);
            cont = ('\0' != pipestr[a + b + 1]);
            pipestr[a] = pipestr[a + b + 1] = '\0';

            fprintf( stderr, "    Setting parameter `%s' to `%s'\n",
                     pipestr, &pipestr[a + 1] );
            filt->setParam( pipestr, &pipestr[a + 1] );
            pipestr += a + b + 2;
        }

        pipeline.add( filt );
    }

    fputs( "Pipeline built successfully\n\n", stderr );
}

int
main(
    int argc,
    char* argv[]
    )
{
    if( argc < 3 || !(argc % 2) )
    {
        fprintf( stderr, "Usage: %s <backend> <pipeline> [<infile> <outfile> ...]\n",
                 argv[0] );
        return EXIT_FAILURE;
    }

    try
    {
        fprintf( stderr, "Initializing backend `%s'\n", argv[1] );
        ImageBackend backend( argv[1] );

        ImageFilterPipeline pipeline;
        buildPipeline( argv[2], backend, pipeline );

        argc -= 3;
        argv += 3;
        for(; 0 < argc; argc -= 2, argv += 2 )
        {
            Image image( backend, argv[0] );
            fprintf( stderr,
                     "Loaded image from file `%s'\n"
                     "Input image info:\n"
                     "  Width:      %u px\n"
                     "  Height:     %u px\n"
                     "  Pixel size: %u B\n",
                     argv[0],
                     image.columns(),
                     image.rows(),
                     image.pixelSize() );

            pipeline.filter( image );

            image.save( argv[1] );
            fprintf( stderr,
                     "Output image info:\n"
                     "  Width:      %u px\n"
                     "  Height:     %u px\n"
                     "  Pixel size: %u B\n"
                     "Saved image to file `%s'\n\n",
                     image.columns(),
                     image.rows(),
                     image.pixelSize(),
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
