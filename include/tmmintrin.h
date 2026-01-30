/* CCC compiler bundled tmmintrin.h - SSSE3 intrinsics */
#ifndef _TMMINTRIN_H_INCLUDED
#define _TMMINTRIN_H_INCLUDED

#include <emmintrin.h>

/* _mm_abs_epi16: absolute value of signed 16-bit integers (PABSW) */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_abs_epi16(__m128i __a)
{
    short *__pa = (short *)&__a;
    __m128i __r;
    short *__pr = (short *)&__r;
    for (int __i = 0; __i < 8; __i++)
        __pr[__i] = __pa[__i] < 0 ? (short)-__pa[__i] : __pa[__i];
    return __r;
}

#endif /* _TMMINTRIN_H_INCLUDED */
