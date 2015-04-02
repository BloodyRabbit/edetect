/** @file
 * @brief Declaration of CudaIntFloatFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_INT_FLOAT_FILTER_HXX__INCL__
#define CUDA__CUDA_INT_FLOAT_FILTER_HXX__INCL__

#include "cuda/CudaFilter.hxx"
#include "cuda/CudaImage.hxx"

/**
 * @brief Converts integer-pixel images to
 *   float-pixel and vice versa.
 *
 * @author Jan Bobek
 */
class CudaIntFloatFilter
: public CudaFilter
{
public:
    /**
     * @brief Applies the conversion to an image.
     *
     * @param[in,out] image
     *   The image to convert.
     */
    void process( CudaImage& image );

protected:
    /// Table of target formats by format
    static const CudaImage::Format FMT_TARGET[];
};

#endif /* !CUDA__CUDA_INT_FLOAT_FILTER_HXX__INCL__ */
