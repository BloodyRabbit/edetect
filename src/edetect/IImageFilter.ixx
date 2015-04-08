/** @file
 * @brief Inline definition of class IImageFilter.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

/*************************************************************************/
/* IImageFilter                                                          */
/*************************************************************************/
inline
IImageFilter::~IImageFilter()
{
}

inline
void
IImageFilter::setParam(
    const char*,
    const void*
    )
{
    throw std::invalid_argument(
        "IImageFilter: Parameter not implemented" );
}
