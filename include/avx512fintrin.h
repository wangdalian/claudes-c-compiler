/* CCC compiler bundled avx512fintrin.h - AVX-512 Foundation intrinsics */
#ifndef _AVX512FINTRIN_H_INCLUDED
#define _AVX512FINTRIN_H_INCLUDED

#include <avx2intrin.h>

/* AVX-512 512-bit vector types */
typedef struct __attribute__((__aligned__(64))) {
    long long __val[8];
} __m512i;

typedef struct __attribute__((__aligned__(64))) {
    double __val[8];
} __m512d;

typedef struct __attribute__((__aligned__(64))) {
    float __val[16];
} __m512;

/* Unaligned variants */
typedef struct __attribute__((__aligned__(1))) {
    long long __val[8];
} __m512i_u;

/* AVX-512 mask types */
typedef unsigned char __mmask8;
typedef unsigned short __mmask16;

/* === Load / Store === */

static __inline__ __m512i __attribute__((__always_inline__))
_mm512_loadu_si512(void const *__p)
{
    __m512i __r;
    __builtin_memcpy(&__r, __p, sizeof(__r));
    return __r;
}

static __inline__ void __attribute__((__always_inline__))
_mm512_storeu_si512(void *__p, __m512i __a)
{
    __builtin_memcpy(__p, &__a, sizeof(__a));
}

/* === Set === */

static __inline__ __m512i __attribute__((__always_inline__))
_mm512_setzero_si512(void)
{
    return (__m512i){ { 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL, 0LL } };
}

static __inline__ __m512i __attribute__((__always_inline__))
_mm512_set1_epi64(long long __q)
{
    return (__m512i){ { __q, __q, __q, __q, __q, __q, __q, __q } };
}

/* === Arithmetic === */

static __inline__ __m512i __attribute__((__always_inline__))
_mm512_add_epi64(__m512i __a, __m512i __b)
{
    return (__m512i){ { __a.__val[0] + __b.__val[0],
                        __a.__val[1] + __b.__val[1],
                        __a.__val[2] + __b.__val[2],
                        __a.__val[3] + __b.__val[3],
                        __a.__val[4] + __b.__val[4],
                        __a.__val[5] + __b.__val[5],
                        __a.__val[6] + __b.__val[6],
                        __a.__val[7] + __b.__val[7] } };
}

/* === Population count === */

/* _mm512_popcnt_epi64: population count for each 64-bit element */
static __inline__ __m512i __attribute__((__always_inline__))
_mm512_popcnt_epi64(__m512i __a)
{
    __m512i __r;
    for (int __i = 0; __i < 8; __i++) {
        unsigned long long __v = (unsigned long long)__a.__val[__i];
        int __cnt = 0;
        while (__v) {
            __cnt++;
            __v &= __v - 1;
        }
        __r.__val[__i] = __cnt;
    }
    return __r;
}

/* === Reduce === */

/* _mm512_reduce_add_epi64: horizontal sum of all 64-bit elements */
static __inline__ long long __attribute__((__always_inline__))
_mm512_reduce_add_epi64(__m512i __a)
{
    return __a.__val[0] + __a.__val[1] + __a.__val[2] + __a.__val[3]
         + __a.__val[4] + __a.__val[5] + __a.__val[6] + __a.__val[7];
}

/* === Float Load / Store === */

static __inline__ __m512 __attribute__((__always_inline__))
_mm512_loadu_ps(void const *__p)
{
    __m512 __r;
    const float *__fp = (const float *)__p;
    for (int __i = 0; __i < 16; __i++)
        __r.__val[__i] = __fp[__i];
    return __r;
}

static __inline__ void __attribute__((__always_inline__))
_mm512_storeu_ps(void *__p, __m512 __a)
{
    float *__fp = (float *)__p;
    for (int __i = 0; __i < 16; __i++)
        __fp[__i] = __a.__val[__i];
}

/* === Float Set === */

static __inline__ __m512 __attribute__((__always_inline__))
_mm512_setzero_ps(void)
{
    __m512 __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__val[__i] = 0.0f;
    return __r;
}

/* === Float Arithmetic === */

static __inline__ __m512 __attribute__((__always_inline__))
_mm512_add_ps(__m512 __a, __m512 __b)
{
    __m512 __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__val[__i] = __a.__val[__i] + __b.__val[__i];
    return __r;
}

static __inline__ __m512 __attribute__((__always_inline__))
_mm512_mul_ps(__m512 __a, __m512 __b)
{
    __m512 __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__val[__i] = __a.__val[__i] * __b.__val[__i];
    return __r;
}

/* _mm512_fmadd_ps: a*b + c (single-precision, 512-bit) */
static __inline__ __m512 __attribute__((__always_inline__))
_mm512_fmadd_ps(__m512 __a, __m512 __b, __m512 __c)
{
    __m512 __r;
    for (int __i = 0; __i < 16; __i++)
        __r.__val[__i] = __a.__val[__i] * __b.__val[__i] + __c.__val[__i];
    return __r;
}

/* === Float Reduce === */

/* _mm512_reduce_add_ps: horizontal sum of all 16 float elements */
static __inline__ float __attribute__((__always_inline__))
_mm512_reduce_add_ps(__m512 __a)
{
    float __sum = 0.0f;
    for (int __i = 0; __i < 16; __i++)
        __sum += __a.__val[__i];
    return __sum;
}

#endif /* _AVX512FINTRIN_H_INCLUDED */
