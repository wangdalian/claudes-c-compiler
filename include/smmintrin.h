/* CCC compiler bundled smmintrin.h - SSE4.1 / SSE4.2 intrinsics */
#ifndef _SMMINTRIN_H_INCLUDED
#define _SMMINTRIN_H_INCLUDED

#include <tmmintrin.h>

/* === SSE4.1 insert/extract intrinsics === */

/* _mm_extract_epi8: extract 8-bit int from lane (PEXTRB) */
#define _mm_extract_epi8(a, imm) \
    ((int)(unsigned char)__builtin_ia32_pextrb128((a), (imm)))

/* _mm_extract_epi32: extract 32-bit int from lane (PEXTRD) */
#define _mm_extract_epi32(a, imm) \
    ((int)__builtin_ia32_pextrd128((a), (imm)))

/* _mm_extract_epi64: extract 64-bit int from lane (PEXTRQ) */
#define _mm_extract_epi64(a, imm) \
    ((long long)__builtin_ia32_pextrq128((a), (imm)))

/* _mm_insert_epi8: insert 8-bit int at lane (PINSRB) */
#define _mm_insert_epi8(a, i, imm) \
    __CCC_M128I_FROM_BUILTIN(__builtin_ia32_pinsrb128((a), (i), (imm)))

/* _mm_insert_epi32: insert 32-bit int at lane (PINSRD) */
#define _mm_insert_epi32(a, i, imm) \
    __CCC_M128I_FROM_BUILTIN(__builtin_ia32_pinsrd128((a), (i), (imm)))

/* _mm_insert_epi64: insert 64-bit int at lane (PINSRQ) */
#define _mm_insert_epi64(a, i, imm) \
    __CCC_M128I_FROM_BUILTIN(__builtin_ia32_pinsrq128((a), (i), (imm)))

/* === SSE4.1 blending === */

/* _mm_blendv_epi8: byte-level blend using mask high bits (PBLENDVB) */
static __inline__ __m128i __attribute__((__always_inline__))
_mm_blendv_epi8(__m128i __a, __m128i __b, __m128i __mask)
{
    /* For each byte: result = (mask_byte & 0x80) ? b_byte : a_byte */
    unsigned char *__pa = (unsigned char *)&__a;
    unsigned char *__pb = (unsigned char *)&__b;
    unsigned char *__pm = (unsigned char *)&__mask;
    __m128i __r;
    unsigned char *__pr = (unsigned char *)&__r;
    for (int __i = 0; __i < 16; __i++)
        __pr[__i] = (__pm[__i] & 0x80) ? __pb[__i] : __pa[__i];
    return __r;
}

/* === CRC32 intrinsics (SSE4.2) === */

static __inline__ unsigned int __attribute__((__always_inline__))
_mm_crc32_u8(unsigned int __crc, unsigned char __v)
{
    return __builtin_ia32_crc32qi(__crc, __v);
}

static __inline__ unsigned int __attribute__((__always_inline__))
_mm_crc32_u16(unsigned int __crc, unsigned short __v)
{
    return __builtin_ia32_crc32hi(__crc, __v);
}

static __inline__ unsigned int __attribute__((__always_inline__))
_mm_crc32_u32(unsigned int __crc, unsigned int __v)
{
    return __builtin_ia32_crc32si(__crc, __v);
}

static __inline__ unsigned long long __attribute__((__always_inline__))
_mm_crc32_u64(unsigned long long __crc, unsigned long long __v)
{
    return __builtin_ia32_crc32di(__crc, __v);
}

#endif /* _SMMINTRIN_H_INCLUDED */
