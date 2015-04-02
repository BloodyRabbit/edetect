/** @file
 * @brief Declaration of CudaDesaturateFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__
#define CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__

#include "cuda/CudaFilter.hxx"

/**
 * @brief Desaturates the image (converts
 *   it to grayscale).
 *
 * @author Jan Bobek
 */
class CudaDesaturateFilter
: public CudaFilter
{
public:
    /**
     * @brief Describes supported methods of desaturation.
     *
     * @author Jan Bobek
     */
    enum Method
    {
        METHOD_AVERAGE,    ///< The Average method.
        METHOD_LIGHTNESS,  ///< The Lightness method.
        METHOD_LUMINOSITY, ///< The Luminosity method.
    };

    /**
     * @brief Initializes the filter.
     *
     * @param[in] method
     *   The chosen method.
     */
    CudaDesaturateFilter( Method method );

    /**
     * @brief Desaturates the image.
     *
     * @param[in,out] image
     *   The image to desaturate.
     */
    void process( CudaImage& image );

protected:
    /// The chosen method.
    Method mMethod;
};

#endif /* !CUDA__CUDA_DESATURATE_FILTER_HXX__INCL__ */
