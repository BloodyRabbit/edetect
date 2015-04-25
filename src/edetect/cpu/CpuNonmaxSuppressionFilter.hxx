/** @file
 * @brief Declaration of class CpuNonmaxSuppressionFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#ifndef CPU__CPU_NONMAX_SUPPRESSION_FILTER_HXX__INCL__
#define CPU__CPU_NONMAX_SUPPRESSION_FILTER_HXX__INCL__

#include "filters/INonmaxSuppressionFilter.hxx"

/**
 * @brief CPU-backed non-maximum suppression filter.
 *
 * @author Jan Bobek
 */
class CpuNonmaxSuppressionFilter
: public INonmaxSuppressionFilter
{
public:
    /// @copydoc INonmaxSuppressionFilter::INonmaxSuppressionFilter(IImageFilter*, IImageFilter*)
    CpuNonmaxSuppressionFilter(
        IImageFilter* first = NULL,
        IImageFilter* second = NULL
        );

protected:
    /// @copydoc INonmaxSuppressionFilter::nonmaxSuppress(IImage&, const IImage&, const IImage&)
    void nonmaxSuppress( IImage& dest, const IImage& first, const IImage& second );
};

#endif /* !CPU__CPU_NONMAX_SUPPRESSION_FILTER_HXX__INCL__ */
