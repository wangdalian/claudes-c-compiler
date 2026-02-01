/* CCC compiler bundled xmmintrin.h - SSE intrinsics */
#ifndef _XMMINTRIN_H_INCLUDED
#define _XMMINTRIN_H_INCLUDED

/* SSE intrinsics are only available on x86/x86-64 targets */
#if !defined(__x86_64__) && !defined(__i386__) && !defined(__i686__)
#error "SSE intrinsics (xmmintrin.h) require an x86 target"
#endif

#include <mmintrin.h>

typedef struct __attribute__((__aligned__(16))) {
    float __val[4];
} __m128;

/* Internal vector type referenced by GCC system headers.
 * Note: vector_size attribute is parsed but vectors are lowered as aggregates. */
typedef float __v4sf __attribute__ ((__vector_size__ (16)));

/* _MM_SHUFFLE: build an immediate for _mm_shuffle_ps / _mm_shuffle_epi32.
 * The result encodes four 2-bit lane selectors as (z<<6|y<<4|x<<2|w). */
#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))

/* === Set / Broadcast === */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_setzero_ps(void)
{
    return (__m128){ { 0.0f, 0.0f, 0.0f, 0.0f } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_set1_ps(float __w)
{
    return (__m128){ { __w, __w, __w, __w } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_set_ps(float __z, float __y, float __x, float __w)
{
    return (__m128){ { __w, __x, __y, __z } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_setr_ps(float __w, float __x, float __y, float __z)
{
    return (__m128){ { __w, __x, __y, __z } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_set_ss(float __w)
{
    return (__m128){ { __w, 0.0f, 0.0f, 0.0f } };
}

/* _mm_set_ps1 is a standard alias for _mm_set1_ps */
#define _mm_set_ps1(w) _mm_set1_ps(w)

/* === Load === */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_loadu_ps(const float *__p)
{
    __m128 __r;
    __builtin_memcpy(&__r, __p, 16);
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_load_ps(const float *__p)
{
    return *(const __m128 *)__p;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_load_ss(const float *__p)
{
    return (__m128){ { *__p, 0.0f, 0.0f, 0.0f } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_load1_ps(const float *__p)
{
    float __v = *__p;
    return (__m128){ { __v, __v, __v, __v } };
}

#define _mm_load_ps1(p) _mm_load1_ps(p)

/* === Store === */

static __inline__ void __attribute__((__always_inline__))
_mm_storeu_ps(float *__p, __m128 __a)
{
    __builtin_memcpy(__p, &__a, 16);
}

static __inline__ void __attribute__((__always_inline__))
_mm_store_ps(float *__p, __m128 __a)
{
    *((__m128 *)__p) = __a;
}

static __inline__ void __attribute__((__always_inline__))
_mm_store_ss(float *__p, __m128 __a)
{
    *__p = __a.__val[0];
}

static __inline__ void __attribute__((__always_inline__))
_mm_store1_ps(float *__p, __m128 __a)
{
    __p[0] = __a.__val[0]; __p[1] = __a.__val[0];
    __p[2] = __a.__val[0]; __p[3] = __a.__val[0];
}

#define _mm_store_ps1(p, a) _mm_store1_ps(p, a)

/* Store the lower 2 floats of __m128 to __m64* memory location. */
static __inline__ void __attribute__((__always_inline__))
_mm_storel_pi(__m64 *__p, __m128 __a)
{
    __builtin_memcpy(__p, &__a, 8);
}

/* Store the upper 2 floats of __m128 to __m64* memory location. */
static __inline__ void __attribute__((__always_inline__))
_mm_storeh_pi(__m64 *__p, __m128 __a)
{
    __builtin_memcpy(__p, (const char *)&__a + 8, 8);
}

/* === Arithmetic === */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_add_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __a.__val[0] + __b.__val[0], __a.__val[1] + __b.__val[1],
                       __a.__val[2] + __b.__val[2], __a.__val[3] + __b.__val[3] } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_sub_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __a.__val[0] - __b.__val[0], __a.__val[1] - __b.__val[1],
                       __a.__val[2] - __b.__val[2], __a.__val[3] - __b.__val[3] } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_mul_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __a.__val[0] * __b.__val[0], __a.__val[1] * __b.__val[1],
                       __a.__val[2] * __b.__val[2], __a.__val[3] * __b.__val[3] } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_div_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __a.__val[0] / __b.__val[0], __a.__val[1] / __b.__val[1],
                       __a.__val[2] / __b.__val[2], __a.__val[3] / __b.__val[3] } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_min_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __a.__val[0] < __b.__val[0] ? __a.__val[0] : __b.__val[0],
                       __a.__val[1] < __b.__val[1] ? __a.__val[1] : __b.__val[1],
                       __a.__val[2] < __b.__val[2] ? __a.__val[2] : __b.__val[2],
                       __a.__val[3] < __b.__val[3] ? __a.__val[3] : __b.__val[3] } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_max_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __a.__val[0] > __b.__val[0] ? __a.__val[0] : __b.__val[0],
                       __a.__val[1] > __b.__val[1] ? __a.__val[1] : __b.__val[1],
                       __a.__val[2] > __b.__val[2] ? __a.__val[2] : __b.__val[2],
                       __a.__val[3] > __b.__val[3] ? __a.__val[3] : __b.__val[3] } };
}

/* Scalar operations (lowest element only, rest pass through __a) */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_add_ss(__m128 __a, __m128 __b)
{
    __a.__val[0] += __b.__val[0];
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_sub_ss(__m128 __a, __m128 __b)
{
    __a.__val[0] -= __b.__val[0];
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_mul_ss(__m128 __a, __m128 __b)
{
    __a.__val[0] *= __b.__val[0];
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_div_ss(__m128 __a, __m128 __b)
{
    __a.__val[0] /= __b.__val[0];
    return __a;
}

/* === Bitwise (float domain) === */
/* These operate on the bitwise representation of float values,
   using memcpy to type-pun between float and unsigned int. */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_and_ps(__m128 __a, __m128 __b)
{
    unsigned int __ai[4], __bi[4];
    __builtin_memcpy(__ai, &__a, 16);
    __builtin_memcpy(__bi, &__b, 16);
    __ai[0] &= __bi[0]; __ai[1] &= __bi[1];
    __ai[2] &= __bi[2]; __ai[3] &= __bi[3];
    __m128 __r;
    __builtin_memcpy(&__r, __ai, 16);
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_andnot_ps(__m128 __a, __m128 __b)
{
    unsigned int __ai[4], __bi[4];
    __builtin_memcpy(__ai, &__a, 16);
    __builtin_memcpy(__bi, &__b, 16);
    __ai[0] = ~__ai[0] & __bi[0]; __ai[1] = ~__ai[1] & __bi[1];
    __ai[2] = ~__ai[2] & __bi[2]; __ai[3] = ~__ai[3] & __bi[3];
    __m128 __r;
    __builtin_memcpy(&__r, __ai, 16);
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_or_ps(__m128 __a, __m128 __b)
{
    unsigned int __ai[4], __bi[4];
    __builtin_memcpy(__ai, &__a, 16);
    __builtin_memcpy(__bi, &__b, 16);
    __ai[0] |= __bi[0]; __ai[1] |= __bi[1];
    __ai[2] |= __bi[2]; __ai[3] |= __bi[3];
    __m128 __r;
    __builtin_memcpy(&__r, __ai, 16);
    return __r;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_xor_ps(__m128 __a, __m128 __b)
{
    unsigned int __ai[4], __bi[4];
    __builtin_memcpy(__ai, &__a, 16);
    __builtin_memcpy(__bi, &__b, 16);
    __ai[0] ^= __bi[0]; __ai[1] ^= __bi[1];
    __ai[2] ^= __bi[2]; __ai[3] ^= __bi[3];
    __m128 __r;
    __builtin_memcpy(&__r, __ai, 16);
    return __r;
}

/* === Square root, Reciprocal, Reciprocal square root === */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_sqrt_ps(__m128 __a)
{
    return (__m128){ { __builtin_sqrtf(__a.__val[0]), __builtin_sqrtf(__a.__val[1]),
                       __builtin_sqrtf(__a.__val[2]), __builtin_sqrtf(__a.__val[3]) } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_sqrt_ss(__m128 __a)
{
    __a.__val[0] = __builtin_sqrtf(__a.__val[0]);
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_rcp_ps(__m128 __a)
{
    return (__m128){ { 1.0f / __a.__val[0], 1.0f / __a.__val[1],
                       1.0f / __a.__val[2], 1.0f / __a.__val[3] } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_rcp_ss(__m128 __a)
{
    __a.__val[0] = 1.0f / __a.__val[0];
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_rsqrt_ps(__m128 __a)
{
    return (__m128){ { 1.0f / __builtin_sqrtf(__a.__val[0]), 1.0f / __builtin_sqrtf(__a.__val[1]),
                       1.0f / __builtin_sqrtf(__a.__val[2]), 1.0f / __builtin_sqrtf(__a.__val[3]) } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_rsqrt_ss(__m128 __a)
{
    __a.__val[0] = 1.0f / __builtin_sqrtf(__a.__val[0]);
    return __a;
}

/* === Comparison (packed) - return all-ones or all-zeros per lane === */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpeq_ps(__m128 __a, __m128 __b)
{
    unsigned int __r[4];
    __r[0] = __a.__val[0] == __b.__val[0] ? 0xFFFFFFFFu : 0;
    __r[1] = __a.__val[1] == __b.__val[1] ? 0xFFFFFFFFu : 0;
    __r[2] = __a.__val[2] == __b.__val[2] ? 0xFFFFFFFFu : 0;
    __r[3] = __a.__val[3] == __b.__val[3] ? 0xFFFFFFFFu : 0;
    __m128 __rv;
    __builtin_memcpy(&__rv, __r, 16);
    return __rv;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmplt_ps(__m128 __a, __m128 __b)
{
    unsigned int __r[4];
    __r[0] = __a.__val[0] < __b.__val[0] ? 0xFFFFFFFFu : 0;
    __r[1] = __a.__val[1] < __b.__val[1] ? 0xFFFFFFFFu : 0;
    __r[2] = __a.__val[2] < __b.__val[2] ? 0xFFFFFFFFu : 0;
    __r[3] = __a.__val[3] < __b.__val[3] ? 0xFFFFFFFFu : 0;
    __m128 __rv;
    __builtin_memcpy(&__rv, __r, 16);
    return __rv;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmple_ps(__m128 __a, __m128 __b)
{
    unsigned int __r[4];
    __r[0] = __a.__val[0] <= __b.__val[0] ? 0xFFFFFFFFu : 0;
    __r[1] = __a.__val[1] <= __b.__val[1] ? 0xFFFFFFFFu : 0;
    __r[2] = __a.__val[2] <= __b.__val[2] ? 0xFFFFFFFFu : 0;
    __r[3] = __a.__val[3] <= __b.__val[3] ? 0xFFFFFFFFu : 0;
    __m128 __rv;
    __builtin_memcpy(&__rv, __r, 16);
    return __rv;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpgt_ps(__m128 __a, __m128 __b)
{
    return _mm_cmplt_ps(__b, __a);
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpge_ps(__m128 __a, __m128 __b)
{
    return _mm_cmple_ps(__b, __a);
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpneq_ps(__m128 __a, __m128 __b)
{
    unsigned int __r[4];
    __r[0] = __a.__val[0] != __b.__val[0] ? 0xFFFFFFFFu : 0;
    __r[1] = __a.__val[1] != __b.__val[1] ? 0xFFFFFFFFu : 0;
    __r[2] = __a.__val[2] != __b.__val[2] ? 0xFFFFFFFFu : 0;
    __r[3] = __a.__val[3] != __b.__val[3] ? 0xFFFFFFFFu : 0;
    __m128 __rv;
    __builtin_memcpy(&__rv, __r, 16);
    return __rv;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpord_ps(__m128 __a, __m128 __b)
{
    unsigned int __r[4];
    __r[0] = (__a.__val[0] == __a.__val[0] && __b.__val[0] == __b.__val[0]) ? 0xFFFFFFFFu : 0;
    __r[1] = (__a.__val[1] == __a.__val[1] && __b.__val[1] == __b.__val[1]) ? 0xFFFFFFFFu : 0;
    __r[2] = (__a.__val[2] == __a.__val[2] && __b.__val[2] == __b.__val[2]) ? 0xFFFFFFFFu : 0;
    __r[3] = (__a.__val[3] == __a.__val[3] && __b.__val[3] == __b.__val[3]) ? 0xFFFFFFFFu : 0;
    __m128 __rv;
    __builtin_memcpy(&__rv, __r, 16);
    return __rv;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpunord_ps(__m128 __a, __m128 __b)
{
    unsigned int __r[4];
    __r[0] = (__a.__val[0] != __a.__val[0] || __b.__val[0] != __b.__val[0]) ? 0xFFFFFFFFu : 0;
    __r[1] = (__a.__val[1] != __a.__val[1] || __b.__val[1] != __b.__val[1]) ? 0xFFFFFFFFu : 0;
    __r[2] = (__a.__val[2] != __a.__val[2] || __b.__val[2] != __b.__val[2]) ? 0xFFFFFFFFu : 0;
    __r[3] = (__a.__val[3] != __a.__val[3] || __b.__val[3] != __b.__val[3]) ? 0xFFFFFFFFu : 0;
    __m128 __rv;
    __builtin_memcpy(&__rv, __r, 16);
    return __rv;
}

/* Scalar comparison intrinsics (operate on element 0 only, rest pass through __a) */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpeq_ss(__m128 __a, __m128 __b)
{
    unsigned int __u = (__a.__val[0] == __b.__val[0]) ? 0xFFFFFFFFu : 0;
    __builtin_memcpy(&__a.__val[0], &__u, 4);
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmplt_ss(__m128 __a, __m128 __b)
{
    unsigned int __u = (__a.__val[0] < __b.__val[0]) ? 0xFFFFFFFFu : 0;
    __builtin_memcpy(&__a.__val[0], &__u, 4);
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmple_ss(__m128 __a, __m128 __b)
{
    unsigned int __u = (__a.__val[0] <= __b.__val[0]) ? 0xFFFFFFFFu : 0;
    __builtin_memcpy(&__a.__val[0], &__u, 4);
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpgt_ss(__m128 __a, __m128 __b)
{
    unsigned int __u = (__a.__val[0] > __b.__val[0]) ? 0xFFFFFFFFu : 0;
    __builtin_memcpy(&__a.__val[0], &__u, 4);
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpge_ss(__m128 __a, __m128 __b)
{
    unsigned int __u = (__a.__val[0] >= __b.__val[0]) ? 0xFFFFFFFFu : 0;
    __builtin_memcpy(&__a.__val[0], &__u, 4);
    return __a;
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cmpneq_ss(__m128 __a, __m128 __b)
{
    unsigned int __u = (__a.__val[0] != __b.__val[0]) ? 0xFFFFFFFFu : 0;
    __builtin_memcpy(&__a.__val[0], &__u, 4);
    return __a;
}

/* === Integer conversion === */

/* TODO: _mm_cvtss_si32 should use current MXCSR rounding mode (round-to-nearest
   by default), but we use C cast truncation for simplicity. This matches
   _mm_cvttss_si32 behavior. */
static __inline__ int __attribute__((__always_inline__))
_mm_cvtss_si32(__m128 __a)
{
    return (int)__a.__val[0];
}

/* Alias: _mm_cvt_ss2si is standard alias for _mm_cvtss_si32 */
#define _mm_cvt_ss2si(a) _mm_cvtss_si32(a)

static __inline__ int __attribute__((__always_inline__))
_mm_cvttss_si32(__m128 __a)
{
    return (int)__a.__val[0];
}

/* Alias: _mm_cvtt_ss2si */
#define _mm_cvtt_ss2si(a) _mm_cvttss_si32(a)

static __inline__ __m128 __attribute__((__always_inline__))
_mm_cvtsi32_ss(__m128 __a, int __b)
{
    __a.__val[0] = (float)__b;
    return __a;
}

/* Alias: _mm_cvt_si2ss */
#define _mm_cvt_si2ss(a, b) _mm_cvtsi32_ss(a, b)

/* === Shuffle === */

/* _mm_shuffle_ps: shuffle floats from __a and __b using immediate mask.
 * Bits [1:0] select from __a for element 0, [3:2] for element 1,
 * [5:4] select from __b for element 2, [7:6] for element 3. */
#define _mm_shuffle_ps(__a, __b, __imm) __extension__ ({ \
    __m128 __r; \
    __r.__val[0] = (__a).__val[(__imm) & 3]; \
    __r.__val[1] = (__a).__val[((__imm) >> 2) & 3]; \
    __r.__val[2] = (__b).__val[((__imm) >> 4) & 3]; \
    __r.__val[3] = (__b).__val[((__imm) >> 6) & 3]; \
    __r; \
})

/* === Unpack / Interleave === */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_unpacklo_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __a.__val[0], __b.__val[0], __a.__val[1], __b.__val[1] } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_unpackhi_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __a.__val[2], __b.__val[2], __a.__val[3], __b.__val[3] } };
}

/* === Move === */

static __inline__ __m128 __attribute__((__always_inline__))
_mm_movehl_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __b.__val[2], __b.__val[3], __a.__val[2], __a.__val[3] } };
}

static __inline__ __m128 __attribute__((__always_inline__))
_mm_movelh_ps(__m128 __a, __m128 __b)
{
    return (__m128){ { __a.__val[0], __a.__val[1], __b.__val[0], __b.__val[1] } };
}

static __inline__ float __attribute__((__always_inline__))
_mm_cvtss_f32(__m128 __a)
{
    return __a.__val[0];
}

/* === Compare (packed) - return all-ones or all-zeros per lane === */

static __inline__ int __attribute__((__always_inline__))
_mm_movemask_ps(__m128 __a)
{
    int __r = 0;
    unsigned int __u;
    __builtin_memcpy(&__u, &__a.__val[0], 4); __r |= (__u >> 31);
    __builtin_memcpy(&__u, &__a.__val[1], 4); __r |= ((__u >> 31) << 1);
    __builtin_memcpy(&__u, &__a.__val[2], 4); __r |= ((__u >> 31) << 2);
    __builtin_memcpy(&__u, &__a.__val[3], 4); __r |= ((__u >> 31) << 3);
    return __r;
}

/* === Prefetch === */

/* Prefetch hint constants */
#define _MM_HINT_T0  3
#define _MM_HINT_T1  2
#define _MM_HINT_T2  1
#define _MM_HINT_NTA 0

/* _mm_prefetch: hint to prefetch data into cache.
 * In our implementation this is a no-op since we don't emit prefetch
 * instructions, but it must be defined for source compatibility. */
#define _mm_prefetch(P, I) ((void)(P), (void)(I))

/* === Aligned memory allocation === */

static __inline__ void *__attribute__((__always_inline__))
_mm_malloc(unsigned long __size, unsigned long __align)
{
    void *__ptr;
    if (__align <= sizeof(void *))
        return __builtin_malloc(__size);
    /* Use posix_memalign for aligned allocation */
    if (__size == 0)
        return (void *)0;
    /* Manually align: allocate extra space for alignment and store original pointer */
    void *__raw = __builtin_malloc(__size + __align + sizeof(void *));
    if (!__raw)
        return (void *)0;
    __ptr = (void *)(((unsigned long)((char *)__raw + sizeof(void *) + __align - 1)) & ~(__align - 1));
    ((void **)__ptr)[-1] = __raw;
    return __ptr;
}

static __inline__ void __attribute__((__always_inline__))
_mm_free(void *__ptr)
{
    if (__ptr)
        __builtin_free(((void **)__ptr)[-1]);
}

/* === Fence === */

static __inline__ void __attribute__((__always_inline__))
_mm_sfence(void)
{
    __builtin_ia32_sfence();
}

static __inline__ void __attribute__((__always_inline__))
_mm_pause(void)
{
    __builtin_ia32_pause();
}

#endif /* _XMMINTRIN_H_INCLUDED */
