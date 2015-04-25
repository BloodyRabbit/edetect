/** @file
 * @brief Declaration of class CudaNonmaxSuppressionFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#ifndef CUDA__CUDA_NONMAX_SUPPRESSION_FILTER_HXX__INCL__
#define CUDA__CUDA_NONMAX_SUPPRESSION_FILTER_HXX__INCL__

#include "filters/INonmaxSuppressionFilter.hxx"

/**
 * @brief CUDA-backed non-maximum suppression filter.
 *
 * @author Jan Bobek
 */
class CudaNonmaxSuppressionFilter
: public INonmaxSuppressionFilter
{
public:
    /// @copydoc INonmaxSuppressionFilter::INonmaxSuppressionFilter(IImageFilter*, IImageFilter*)
    CudaNonmaxSuppressionFilter(
        IImageFilter* first = NULL,
        IImageFilter* second = NULL
        );

protected:
    /// @copydoc INonmaxSuppressionFilter::nonmaxSuppress(IImage&, const IImage&, const IImage&)
    void nonmaxSuppress( IImage& dest, const IImage& first, const IImage& second );
};

#endif /* !CUDA__CUDA_NONMAX_SUPPRESSION_FILTER_HXX__INCL__ */
