/** @file
 * @brief Declaration of CudaFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_FILTER_HXX__INCL__
#define CUDA__CUDA_FILTER_HXX__INCL__

class CudaImage;

/**
 * @brief A filter applicable to CudaImage.
 *
 * @author Jan Bobek
 */
class CudaFilter
{
public:
    /**
     * @brief A virtual destructor.
     */
    virtual ~CudaFilter() {}

    /**
     * @brief Applies the filter to an image.
     *
     * @param[in,out] image
     *   The image to filter.
     */
    virtual void process( CudaImage& image ) = 0;
};

#endif /* !CUDA__CUDA_FILTER_HXX__INCL__ */
