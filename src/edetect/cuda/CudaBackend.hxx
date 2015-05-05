/** @file
 * @brief Declaration of class CudaBackend.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#ifndef CUDA__CUDA_BACKEND_HXX__INCL__
#define CUDA__CUDA_BACKEND_HXX__INCL__

#include "IImageBackend.hxx"

/**
 * @brief A CUDA image backend.
 *
 * @author Jan Bobek
 */
class CudaBackend
: public IImageBackend
{
public:
    /**
     * @brief Initializes the CUDA backend.
     */
    CudaBackend();
    /**
     * @brief Deinitializes the CUDA backend.
     */
    ~CudaBackend();

    /**
     * @brief Creates a CUDA-backed image.
     *
     * @return
     *   A CUDA-backed image.
     */
    IImage* createImage();
    /**
     * @brief Creates a CUDA-backed image filter.
     *
     * @param[in] name
     *   Name of the filter.
     *
     * @return
     *   A CUDA-backed filter of given name.
     */
    IImageFilter* createFilter( const char* name );
};

#endif /* !CUDA__CUDA_BACKEND_HXX__INCL__ */
