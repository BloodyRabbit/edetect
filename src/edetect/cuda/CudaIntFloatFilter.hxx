/** @file
 * @brief Declaration of CudaIntFloatFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_INT_FLOAT_FILTER_HXX__INCL__
#define CUDA__CUDA_INT_FLOAT_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Converts integer-pixel images to
 *   float-pixel and vice versa.
 *
 * @author Jan Bobek
 */
class CudaIntFloatFilter
: public IImageFilter
{
public:
    /**
     * @brief Applies the conversion to a CUDA image.
     *
     * @param[in,out] image
     *   The image to convert.
     */
    void filter( CudaImage& image );

protected:
    /// Table of target formats by format
    static const Image::Format FMT_TARGET[];
};

#endif /* !CUDA__CUDA_INT_FLOAT_FILTER_HXX__INCL__ */
