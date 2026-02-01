/* CCC compiler bundled avxintrin.h - AVX intrinsics */
#ifndef _AVXINTRIN_H_INCLUDED
#define _AVXINTRIN_H_INCLUDED

#include <emmintrin.h>

/* AVX 256-bit vector types */
typedef struct __attribute__((__aligned__(32))) {
    float __val[8];
} __m256;

typedef struct __attribute__((__aligned__(32))) {
    double __val[4];
} __m256d;

typedef struct __attribute__((__aligned__(32))) {
    long long __val[4];
} __m256i;

/* Unaligned variants */
typedef struct __attribute__((__aligned__(1))) {
    float __val[8];
} __m256_u;

typedef struct __attribute__((__aligned__(1))) {
    double __val[4];
} __m256d_u;

typedef struct __attribute__((__aligned__(1))) {
    long long __val[4];
} __m256i_u;

/* === Load / Store === */

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_loadu_si256(__m256i_u const *__p)
{
    __m256i __r;
    __builtin_memcpy(&__r, __p, sizeof(__r));
    return __r;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_load_si256(__m256i const *__p)
{
    return *__p;
}

static __inline__ void __attribute__((__always_inline__))
_mm256_storeu_si256(__m256i_u *__p, __m256i __a)
{
    __builtin_memcpy(__p, &__a, sizeof(__a));
}

static __inline__ void __attribute__((__always_inline__))
_mm256_store_si256(__m256i *__p, __m256i __a)
{
    *__p = __a;
}

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_lddqu_si256(__m256i_u const *__p)
{
    __m256i __r;
    __builtin_memcpy(&__r, __p, sizeof(__r));
    return __r;
}

/* Float load/store */
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_loadu_ps(float const *__p)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__val[__i] = __p[__i];
    return __r;
}

static __inline__ void __attribute__((__always_inline__))
_mm256_storeu_ps(float *__p, __m256 __a)
{
    for (int __i = 0; __i < 8; __i++)
        __p[__i] = __a.__val[__i];
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_loadu_pd(double const *__p)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__val[__i] = __p[__i];
    return __r;
}

static __inline__ void __attribute__((__always_inline__))
_mm256_storeu_pd(double *__p, __m256d __a)
{
    for (int __i = 0; __i < 4; __i++)
        __p[__i] = __a.__val[__i];
}

/* === Set === */

static __inline__ __m256i __attribute__((__always_inline__))
_mm256_setzero_si256(void)
{
    return (__m256i){ { 0LL, 0LL, 0LL, 0LL } };
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_setzero_ps(void)
{
    return (__m256){ { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } };
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_setzero_pd(void)
{
    return (__m256d){ { 0.0, 0.0, 0.0, 0.0 } };
}

/* === Cast between 256-bit and 128-bit === */

/* Extract low 128 bits of __m256i as __m128i */
static __inline__ __m128i __attribute__((__always_inline__))
_mm256_castsi256_si128(__m256i __a)
{
    return (__m128i){ { __a.__val[0], __a.__val[1] } };
}

/* Extract low 128 bits of __m256 as __m128 */
static __inline__ __m128 __attribute__((__always_inline__))
_mm256_castps256_ps128(__m256 __a)
{
    return (__m128){ { __a.__val[0], __a.__val[1], __a.__val[2], __a.__val[3] } };
}

/* Zero-extend __m128i to __m256i (upper 128 bits undefined/zero) */
static __inline__ __m256i __attribute__((__always_inline__))
_mm256_castsi128_si256(__m128i __a)
{
    return (__m256i){ { __a.__val[0], __a.__val[1], 0LL, 0LL } };
}

/* Extract 128-bit lane from __m256 (imm must be 0 or 1) */
static __inline__ __m128 __attribute__((__always_inline__))
_mm256_extractf128_ps(__m256 __a, int __imm)
{
    if (__imm & 1)
        return (__m128){ { __a.__val[4], __a.__val[5], __a.__val[6], __a.__val[7] } };
    else
        return (__m128){ { __a.__val[0], __a.__val[1], __a.__val[2], __a.__val[3] } };
}

/* === Float Arithmetic === */

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_add_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__val[__i] = __a.__val[__i] + __b.__val[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_sub_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__val[__i] = __a.__val[__i] - __b.__val[__i];
    return __r;
}

static __inline__ __m256 __attribute__((__always_inline__))
_mm256_mul_ps(__m256 __a, __m256 __b)
{
    __m256 __r;
    for (int __i = 0; __i < 8; __i++)
        __r.__val[__i] = __a.__val[__i] * __b.__val[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_add_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__val[__i] = __a.__val[__i] + __b.__val[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_sub_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__val[__i] = __a.__val[__i] - __b.__val[__i];
    return __r;
}

static __inline__ __m256d __attribute__((__always_inline__))
_mm256_mul_pd(__m256d __a, __m256d __b)
{
    __m256d __r;
    for (int __i = 0; __i < 4; __i++)
        __r.__val[__i] = __a.__val[__i] * __b.__val[__i];
    return __r;
}

#endif /* _AVXINTRIN_H_INCLUDED */
