/** @file
 * @brief Inline definition of class IImage.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

/*************************************************************************/
/* IImage                                                                */
/*************************************************************************/
inline
IImage::IImage()
: mData( NULL ),
  mRows( 0 ),
  mColumns( 0 ),
  mStride( 0 ),
  mFmt( Image::FMT_INVALID )
{
}

inline
IImage::~IImage()
{
}

inline
unsigned char*
IImage::data()
{
    return mData;
}

inline
const unsigned char*
IImage::data() const
{
    return mData;
}

inline
unsigned int
IImage::rows() const
{
    return mRows;
}

inline
unsigned int
IImage::columns() const
{
    return mColumns;
}

inline
unsigned int
IImage::stride() const
{
    return mStride;
}

inline
Image::Format
IImage::format() const
{
    return mFmt;
}

inline
void
IImage::swap(
    IImage& oth
    )
{
    std::swap( mData, oth.mData );
    std::swap( mRows, oth.mRows );
    std::swap( mColumns, oth.mColumns );
    std::swap( mStride, oth.mStride );
    std::swap( mFmt, oth.mFmt );
}
