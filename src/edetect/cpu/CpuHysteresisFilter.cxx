/** @file
 * @brief Definition of class CpuHysteresisFilter.
 *
 * @author Jan Bobek
 * @since 26th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuHysteresisFilter.hxx"

/*************************************************************************/
/* CpuHysteresisFilter                                                   */
/*************************************************************************/
void
CpuHysteresisFilter::hysteresis(
    IImage& dest,
    const IImage& src
    )
{
    pt2d pt;
    std::stack< pt2d > st;

    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        float* const dstp =
            (float*)(dest.data() + row * dest.stride());
        const float* const srcp =
            (const float*)(src.data() + row * src.stride());

        for( unsigned int col = 0; col < src.columns(); ++col )
            if( srcp[col] < mThresholdHigh )
                dstp[col] = 0.0f;
            else
            {
                dstp[col] = 1.0f;
                st.push( pt2d(row, col) );
            }
    }

    while( !st.empty() )
    {
        pt = st.top();
        st.pop();

        enqueue( dest, src, st, pt2d(pt.first - 1, pt.second - 1) );
        enqueue( dest, src, st, pt2d(pt.first - 1, pt.second    ) );
        enqueue( dest, src, st, pt2d(pt.first - 1, pt.second + 1) );

        enqueue( dest, src, st, pt2d(pt.first, pt.second - 1) );
        enqueue( dest, src, st, pt2d(pt.first, pt.second + 1) );

        enqueue( dest, src, st, pt2d(pt.first + 1, pt.second - 1) );
        enqueue( dest, src, st, pt2d(pt.first + 1, pt.second    ) );
        enqueue( dest, src, st, pt2d(pt.first + 1, pt.second + 1) );
    }
}

void
CpuHysteresisFilter::enqueue(
    IImage& dest,
    const IImage& src,
    std::stack< pt2d >& st,
    const pt2d& pt
    )
{
    if( !(pt.first < src.rows() &&
          pt.second < src.columns()) )
        return;

    float* const dstp =
        (float*)(dest.data() + pt.first * dest.stride())
        + pt.second;
    const float* const srcp =
        (const float*)(src.data() + pt.first * src.stride())
        + pt.second;

    if( *srcp < mThresholdLow || 1.0f == *dstp )
        return;

    *dstp = 1.0f;
    st.push( pt );
}
