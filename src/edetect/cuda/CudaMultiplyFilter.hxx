/** @file
 * @brief Declaration of class CudaMultiplyFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#ifndef CUDA_MULTIPLY_FILTER_HXX__INCL__
#define CUDA_MULTIPLY_FILTER_HXX__INCL__

#include "filters/IMultiplyFilter.hxx"

/**
 * @brief CUDA-backed filter which multiplies
 *   one image with another.
 *
 * @author Jan Bobek
 */
class CudaMultiplyFilter
: public IMultiplyFilter
{
public:
    /// @copydoc IMultiplyFilter::IMultiplyFilter(IImageFilter*, IImageFilter*)
    CudaMultiplyFilter(
        IImageFilter* first = NULL,
        IImageFilter* second = NULL
        );

protected:
    /**
     * @brief Multiplies one image with another.
     *
     * @param[in,out] first
     *   Image filtered by the first filter.
     * @param[in] second
     *   Image filtered by the second filter.
     */
    void filter2( IImage& first, const IImage& second );
};

#endif /* !CUDA_MULTIPLY_FILTER_HXX__INCL__ */
