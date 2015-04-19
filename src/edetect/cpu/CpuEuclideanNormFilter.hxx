/** @file
 * @brief Declaration of class CpuEuclideanNormFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#ifndef CPU_EUCLIDEAN_NORM_FILTER_HXX__INCL__
#define CPU_EUCLIDEAN_NORM_FILTER_HXX__INCL__

#include "IEuclideanNormFilter.hxx"

/**
 * @brief CPU-backed filter which computes
 *   the Euclidean norm.
 *
 * @author Jan Bobek
 */
class CpuEuclideanNormFilter
: public IEuclideanNormFilter
{
public:
    /// @copydoc IEuclideanNormFilter::IEuclideanNormFilter(IImageFilter*, IImageFilter*)
    CpuEuclideanNormFilter(
        IImageFilter* first = NULL,
        IImageFilter* second = NULL
        );

protected:
    /**
     * @brief Computes the Euclidean norm.
     *
     * @param[in,out] first
     *   Image filtered by the first filter.
     * @param[in] second
     *   Image filtered by the second filter.
     */
    void filter2( IImage& first, const IImage& second );
};

#endif /* !CPU_EUCLIDEAN_NORM_FILTER_HXX__INCL__ */
