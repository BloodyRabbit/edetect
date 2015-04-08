/** @file
 * @brief Declaration of class CpuBackend.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef CPU__CPU_BACKEND_HXX__INCL__
#define CPU__CPU_BACKEND_HXX__INCL__

#include "IImageBackend.hxx"

/**
 * @brief A CPU backend.
 *
 * @author Jan Bobek
 */
class CpuBackend
: public IImageBackend
{
public:
    /**
     * @brief Creates a CPU-backed image.
     *
     * @return
     *   A CPU-backed image.
     */
    IImage* createImage();
    /**
     * @brief Creates a CPU-backed image filter.
     *
     * @param[in] name
     *   Name of the filter.
     *
     * @return
     *   A CPU-backed image filter.
     */
    IImageFilter* createFilter( const char* name );
};

#endif /* !CPU__CPU_BACKEND_HXX__INCL__ */
