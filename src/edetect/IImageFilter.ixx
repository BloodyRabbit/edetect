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
    const char* name,
    ...
    )
{
    va_list ap;
    va_start( ap, name );
    setParamVa( name, ap );
    va_end( ap );
}

inline
void
IImageFilter::setParamVa(
    const char*,
    va_list
    )
{
    throw std::invalid_argument(
        "IImageFilter: Parameter not implemented" );
}
