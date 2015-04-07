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
IImage::~IImage()
{
}

inline
void
IImage::swap(
    CudaImage&
    )
{
    throw std::invalid_argument(
        "IImage:: Swapping with CudaImage not implemented" );
}

inline
IImage&
IImage::operator=(
    const IImage& oth
    )
{
    oth.duplicate( *this );
    return *this;
}

inline
CudaImage&
IImage::operator=(
    const CudaImage&
    )
{
    throw std::invalid_argument(
        "IImage: Duplicating CudaImage not implemented" );
}
