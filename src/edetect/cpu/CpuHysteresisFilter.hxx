/** @file
 * @brief Declaration of class CpuHysteresisFilter.
 *
 * @author Jan Bobek
 * @since 26th April 2015
 */

#ifndef CPU__CPU_HYSTERESIS_FILTER_HXX__INCL__
#define CPU__CPU_HYSTERESIS_FILTER_HXX__INCL__

#include "filters/IHysteresisFilter.hxx"

/**
 * @brief CPU-backed implementation of
 *   a hysteresis filter.
 *
 * @author Jan Bobek
 */
class CpuHysteresisFilter
: public IHysteresisFilter
{
protected:
    /// Helper typedef of a point for improved readability.
    typedef std::pair< unsigned short, unsigned short > pt2d;

    /// @copydoc IHysteresisFilter::hysteresis(IImage&, const IImage&)
    void hysteresis( IImage& dest, const IImage& src );
    /**
     * @brief Enqueues a point if it matches
     *   hysteresis criteria.
     *
     * @param[in] dest
     *   The destination image.
     * @param[in] src
     *   The source image.
     * @param[in] st
     *   Where to push the point.
     * @param[in] pt
     *   The point in question.
     */
    void enqueue(
        IImage& dest,
        const IImage& src,
        std::stack< pt2d >& st,
        const pt2d& pt
        );
};

#endif /* !CPU__CPU_HYSTERESIS_FILTER_HXX__INCL__ */
