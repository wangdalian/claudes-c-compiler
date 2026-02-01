/* CCC compiler bundled arm_neon.h - ARM NEON intrinsics */
#ifndef _ARM_NEON_H_INCLUDED
#define _ARM_NEON_H_INCLUDED

/* ===== Scalar polynomial types ===== */
typedef unsigned long long poly64_t;
typedef __uint128_t poly128_t;

/* ===== 128-bit vector types (Q registers) ===== */

typedef struct __attribute__((__aligned__(16))) {
    unsigned char __val[16];
} uint8x16_t;

typedef struct __attribute__((__aligned__(16))) {
    signed char __val[16];
} int8x16_t;

typedef struct __attribute__((__aligned__(16))) {
    unsigned short __val[8];
} uint16x8_t;

typedef struct __attribute__((__aligned__(16))) {
    short __val[8];
} int16x8_t;

typedef struct __attribute__((__aligned__(16))) {
    unsigned int __val[4];
} uint32x4_t;

typedef struct __attribute__((__aligned__(16))) {
    int __val[4];
} int32x4_t;

typedef struct __attribute__((__aligned__(16))) {
    unsigned long long __val[2];
} uint64x2_t;

typedef struct __attribute__((__aligned__(16))) {
    long long __val[2];
} int64x2_t;

typedef struct __attribute__((__aligned__(16))) {
    float __val[4];
} float32x4_t;

typedef struct __attribute__((__aligned__(16))) {
    double __val[2];
} float64x2_t;

/* Polynomial 128-bit types */
typedef struct __attribute__((__aligned__(16))) {
    unsigned char __val[16];
} poly8x16_t;

typedef struct __attribute__((__aligned__(16))) {
    unsigned short __val[8];
} poly16x8_t;

typedef struct __attribute__((__aligned__(16))) {
    unsigned long long __val[2];
} poly64x2_t;

/* ===== 64-bit vector types (D registers) ===== */

typedef struct __attribute__((__aligned__(8))) {
    unsigned char __val[8];
} uint8x8_t;

typedef struct __attribute__((__aligned__(8))) {
    signed char __val[8];
} int8x8_t;

typedef struct __attribute__((__aligned__(8))) {
    unsigned short __val[4];
} uint16x4_t;

typedef struct __attribute__((__aligned__(8))) {
    short __val[4];
} int16x4_t;

typedef struct __attribute__((__aligned__(8))) {
    unsigned int __val[2];
} uint32x2_t;

typedef struct __attribute__((__aligned__(8))) {
    int __val[2];
} int32x2_t;

typedef struct __attribute__((__aligned__(8))) {
    unsigned long long __val[1];
} uint64x1_t;

typedef struct __attribute__((__aligned__(8))) {
    long long __val[1];
} int64x1_t;

typedef struct __attribute__((__aligned__(8))) {
    float __val[2];
} float32x2_t;

typedef struct __attribute__((__aligned__(8))) {
    unsigned char __val[8];
} poly8x8_t;

typedef struct __attribute__((__aligned__(8))) {
    unsigned long long __val[1];
} poly64x1_t;

/* ===== Array-of-vectors (structure) types ===== */

/* uint8x8 multi-vector */
typedef struct { uint8x8_t val[2]; } uint8x8x2_t;
typedef struct { uint8x8_t val[3]; } uint8x8x3_t;
typedef struct { uint8x8_t val[4]; } uint8x8x4_t;

/* uint8x16 multi-vector */
typedef struct { uint8x16_t val[2]; } uint8x16x2_t;
typedef struct { uint8x16_t val[3]; } uint8x16x3_t;
typedef struct { uint8x16_t val[4]; } uint8x16x4_t;

/* uint16x4 multi-vector */
typedef struct { uint16x4_t val[2]; } uint16x4x2_t;
typedef struct { uint16x4_t val[4]; } uint16x4x4_t;

/* uint16x8 multi-vector */
typedef struct { uint16x8_t val[2]; } uint16x8x2_t;
typedef struct { uint16x8_t val[4]; } uint16x8x4_t;

/* uint32x2 multi-vector */
typedef struct { uint32x2_t val[2]; } uint32x2x2_t;
typedef struct { uint32x2_t val[4]; } uint32x2x4_t;

/* uint32x4 multi-vector */
typedef struct { uint32x4_t val[2]; } uint32x4x2_t;
typedef struct { uint32x4_t val[4]; } uint32x4x4_t;

/* int8x8 multi-vector */
typedef struct { int8x8_t val[2]; } int8x8x2_t;

/* ================================================================== */
/*                          LOAD INTRINSICS                           */
/* ================================================================== */

/* --- vld1q: load one 128-bit vector --- */

static __inline__ uint8x16_t __attribute__((__always_inline__))
vld1q_u8(const unsigned char *__p)
{
    uint8x16_t __ret;
    __builtin_memcpy(&__ret, __p, 16);
    return __ret;
}

static __inline__ int8x16_t __attribute__((__always_inline__))
vld1q_s8(const signed char *__p)
{
    int8x16_t __ret;
    __builtin_memcpy(&__ret, __p, 16);
    return __ret;
}

static __inline__ uint16x8_t __attribute__((__always_inline__))
vld1q_u16(const unsigned short *__p)
{
    uint16x8_t __ret;
    __builtin_memcpy(&__ret, __p, 16);
    return __ret;
}

static __inline__ uint32x4_t __attribute__((__always_inline__))
vld1q_u32(const unsigned int *__p)
{
    uint32x4_t __ret;
    __builtin_memcpy(&__ret, __p, 16);
    return __ret;
}

static __inline__ uint64x2_t __attribute__((__always_inline__))
vld1q_u64(const unsigned long long *__p)
{
    uint64x2_t __ret;
    __builtin_memcpy(&__ret, __p, 16);
    return __ret;
}

/* --- vld1q_dup: load and broadcast --- */

static __inline__ uint32x4_t __attribute__((__always_inline__))
vld1q_dup_u32(const unsigned int *__p)
{
    uint32x4_t __ret;
    unsigned int __v = *__p;
    __ret.__val[0] = __v;
    __ret.__val[1] = __v;
    __ret.__val[2] = __v;
    __ret.__val[3] = __v;
    return __ret;
}

/* --- vld1q_lane: load one lane --- */

static __inline__ uint32x4_t __attribute__((__always_inline__))
vld1q_lane_u32(const unsigned int *__p, uint32x4_t __v, int __lane)
{
    __v.__val[__lane] = *__p;
    return __v;
}

/* --- vld1q multi-load --- */

static __inline__ uint8x16x4_t __attribute__((__always_inline__))
vld1q_u8_x4(const unsigned char *__p)
{
    uint8x16x4_t __ret;
    __builtin_memcpy(&__ret, __p, 64);
    return __ret;
}

/* --- vld1: load one 64-bit vector --- */

static __inline__ uint8x8_t __attribute__((__always_inline__))
vld1_u8(const unsigned char *__p)
{
    uint8x8_t __ret;
    __builtin_memcpy(&__ret, __p, 8);
    return __ret;
}

/* --- Structure loads (de-interleave) --- */

static __inline__ uint8x16x3_t __attribute__((__always_inline__))
vld3q_u8(const unsigned char *__p)
{
    /* De-interleave 48 bytes into 3 vectors of 16 bytes each */
    uint8x16x3_t __ret;
    for (int __i = 0; __i < 16; __i++) {
        __ret.val[0].__val[__i] = __p[__i * 3 + 0];
        __ret.val[1].__val[__i] = __p[__i * 3 + 1];
        __ret.val[2].__val[__i] = __p[__i * 3 + 2];
    }
    return __ret;
}

static __inline__ uint8x8x3_t __attribute__((__always_inline__))
vld3_dup_u8(const unsigned char *__p)
{
    /* Load 3 bytes and duplicate each into all lanes */
    uint8x8x3_t __ret;
    for (int __i = 0; __i < 8; __i++) {
        __ret.val[0].__val[__i] = __p[0];
        __ret.val[1].__val[__i] = __p[1];
        __ret.val[2].__val[__i] = __p[2];
    }
    return __ret;
}

static __inline__ uint8x8x3_t __attribute__((__always_inline__))
vld3_lane_u8(const unsigned char *__p, uint8x8x3_t __v, int __lane)
{
    __v.val[0].__val[__lane] = __p[0];
    __v.val[1].__val[__lane] = __p[1];
    __v.val[2].__val[__lane] = __p[2];
    return __v;
}

static __inline__ uint32x2x4_t __attribute__((__always_inline__))
vld4_u32(const unsigned int *__p)
{
    /* De-interleave 8 uint32s into 4 vectors of 2 */
    uint32x2x4_t __ret;
    for (int __i = 0; __i < 2; __i++) {
        __ret.val[0].__val[__i] = __p[__i * 4 + 0];
        __ret.val[1].__val[__i] = __p[__i * 4 + 1];
        __ret.val[2].__val[__i] = __p[__i * 4 + 2];
        __ret.val[3].__val[__i] = __p[__i * 4 + 3];
    }
    return __ret;
}

/* ================================================================== */
/*                         STORE INTRINSICS                           */
/* ================================================================== */

static __inline__ void __attribute__((__always_inline__))
vst1q_u8(unsigned char *__p, uint8x16_t __a)
{
    __builtin_memcpy(__p, &__a, 16);
}

static __inline__ void __attribute__((__always_inline__))
vst1q_s8(signed char *__p, int8x16_t __a)
{
    __builtin_memcpy(__p, &__a, 16);
}

static __inline__ void __attribute__((__always_inline__))
vst1q_u16(unsigned short *__p, uint16x8_t __a)
{
    __builtin_memcpy(__p, &__a, 16);
}

static __inline__ void __attribute__((__always_inline__))
vst1q_u32(unsigned int *__p, uint32x4_t __a)
{
    __builtin_memcpy(__p, &__a, 16);
}

static __inline__ void __attribute__((__always_inline__))
vst1q_u64(unsigned long long *__p, uint64x2_t __a)
{
    __builtin_memcpy(__p, &__a, 16);
}

/* --- vst1_lane: store one lane --- */

static __inline__ void __attribute__((__always_inline__))
vst1_lane_u32(unsigned int *__p, uint32x2_t __a, int __lane)
{
    *__p = __a.__val[__lane];
}

/* --- Structure stores (interleave) --- */

static __inline__ void __attribute__((__always_inline__))
vst3_u8(unsigned char *__p, uint8x8x3_t __a)
{
    for (int __i = 0; __i < 8; __i++) {
        __p[__i * 3 + 0] = __a.val[0].__val[__i];
        __p[__i * 3 + 1] = __a.val[1].__val[__i];
        __p[__i * 3 + 2] = __a.val[2].__val[__i];
    }
}

static __inline__ void __attribute__((__always_inline__))
vst4q_u8(unsigned char *__p, uint8x16x4_t __a)
{
    for (int __i = 0; __i < 16; __i++) {
        __p[__i * 4 + 0] = __a.val[0].__val[__i];
        __p[__i * 4 + 1] = __a.val[1].__val[__i];
        __p[__i * 4 + 2] = __a.val[2].__val[__i];
        __p[__i * 4 + 3] = __a.val[3].__val[__i];
    }
}

static __inline__ void __attribute__((__always_inline__))
vst4_lane_u32(unsigned int *__p, uint32x2x4_t __a, int __lane)
{
    __p[0] = __a.val[0].__val[__lane];
    __p[1] = __a.val[1].__val[__lane];
    __p[2] = __a.val[2].__val[__lane];
    __p[3] = __a.val[3].__val[__lane];
}

/* ================================================================== */
/*                       BITWISE OPERATIONS                           */
/* ================================================================== */

static __inline__ uint8x16_t __attribute__((__always_inline__))
veorq_u8(uint8x16_t __a, uint8x16_t __b)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = __a.__val[__i] ^ __b.__val[__i];
    return __ret;
}

static __inline__ uint64x2_t __attribute__((__always_inline__))
veorq_u64(uint64x2_t __a, uint64x2_t __b)
{
    uint64x2_t __ret;
    __ret.__val[0] = __a.__val[0] ^ __b.__val[0];
    __ret.__val[1] = __a.__val[1] ^ __b.__val[1];
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vandq_u8(uint8x16_t __a, uint8x16_t __b)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = __a.__val[__i] & __b.__val[__i];
    return __ret;
}

static __inline__ uint16x8_t __attribute__((__always_inline__))
vandq_u16(uint16x8_t __a, uint16x8_t __b)
{
    uint16x8_t __ret;
    for (int __i = 0; __i < 8; __i++)
        __ret.__val[__i] = __a.__val[__i] & __b.__val[__i];
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vorrq_u8(uint8x16_t __a, uint8x16_t __b)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = __a.__val[__i] | __b.__val[__i];
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vmvnq_u8(uint8x16_t __a)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = ~__a.__val[__i];
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vbicq_u8(uint8x16_t __a, uint8x16_t __b)
{
    /* Bit clear: a & ~b */
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = __a.__val[__i] & ~__b.__val[__i];
    return __ret;
}

/* --- 64-bit bitwise --- */

static __inline__ uint8x8_t __attribute__((__always_inline__))
vbsl_u8(uint8x8_t __sel, uint8x8_t __a, uint8x8_t __b)
{
    /* Bitwise select: (sel & a) | (~sel & b) */
    uint8x8_t __ret;
    for (int __i = 0; __i < 8; __i++)
        __ret.__val[__i] = (__sel.__val[__i] & __a.__val[__i]) |
                           (~__sel.__val[__i] & __b.__val[__i]);
    return __ret;
}

/* ================================================================== */
/*                       SHIFT OPERATIONS                             */
/* ================================================================== */

static __inline__ uint8x16_t __attribute__((__always_inline__))
vshlq_n_u8(uint8x16_t __a, int __n)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = (unsigned char)(__a.__val[__i] << __n);
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vshrq_n_u8(uint8x16_t __a, int __n)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = __a.__val[__i] >> __n;
    return __ret;
}

static __inline__ int8x16_t __attribute__((__always_inline__))
vshrq_n_s8(int8x16_t __a, int __n)
{
    int8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = __a.__val[__i] >> __n;
    return __ret;
}

static __inline__ uint64x2_t __attribute__((__always_inline__))
vshrq_n_u64(uint64x2_t __a, int __n)
{
    uint64x2_t __ret;
    __ret.__val[0] = __a.__val[0] >> __n;
    __ret.__val[1] = __a.__val[1] >> __n;
    return __ret;
}

/* ================================================================== */
/*                    DUPLICATE (BROADCAST)                            */
/* ================================================================== */

static __inline__ uint8x16_t __attribute__((__always_inline__))
vdupq_n_u8(unsigned char __a)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = __a;
    return __ret;
}

static __inline__ int8x16_t __attribute__((__always_inline__))
vdupq_n_s8(signed char __a)
{
    int8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = __a;
    return __ret;
}

static __inline__ uint16x8_t __attribute__((__always_inline__))
vdupq_n_u16(unsigned short __a)
{
    uint16x8_t __ret;
    for (int __i = 0; __i < 8; __i++)
        __ret.__val[__i] = __a;
    return __ret;
}

static __inline__ uint32x4_t __attribute__((__always_inline__))
vdupq_n_u32(unsigned int __a)
{
    uint32x4_t __ret;
    for (int __i = 0; __i < 4; __i++)
        __ret.__val[__i] = __a;
    return __ret;
}

static __inline__ uint8x8_t __attribute__((__always_inline__))
vdup_n_u8(unsigned char __a)
{
    uint8x8_t __ret;
    for (int __i = 0; __i < 8; __i++)
        __ret.__val[__i] = __a;
    return __ret;
}

static __inline__ uint64x1_t __attribute__((__always_inline__))
vmov_n_u64(unsigned long long __a)
{
    uint64x1_t __ret;
    __ret.__val[0] = __a;
    return __ret;
}

/* ================================================================== */
/*                    COMBINE / SPLIT                                  */
/* ================================================================== */

static __inline__ uint8x16_t __attribute__((__always_inline__))
vcombine_u8(uint8x8_t __lo, uint8x8_t __hi)
{
    uint8x16_t __ret;
    __builtin_memcpy(&__ret.__val[0], &__lo.__val[0], 8);
    __builtin_memcpy(&__ret.__val[8], &__hi.__val[0], 8);
    return __ret;
}

static __inline__ int8x16_t __attribute__((__always_inline__))
vcombine_s8(int8x8_t __lo, int8x8_t __hi)
{
    int8x16_t __ret;
    __builtin_memcpy(&__ret.__val[0], &__lo.__val[0], 8);
    __builtin_memcpy(&__ret.__val[8], &__hi.__val[0], 8);
    return __ret;
}

static __inline__ uint64x2_t __attribute__((__always_inline__))
vcombine_u64(uint64x1_t __lo, uint64x1_t __hi)
{
    uint64x2_t __ret;
    __ret.__val[0] = __lo.__val[0];
    __ret.__val[1] = __hi.__val[0];
    return __ret;
}

static __inline__ uint16x8_t __attribute__((__always_inline__))
vcombine_u16(uint16x4_t __lo, uint16x4_t __hi)
{
    uint16x8_t __ret;
    __builtin_memcpy(&__ret.__val[0], &__lo.__val[0], 8);
    __builtin_memcpy(&__ret.__val[4], &__hi.__val[0], 8);
    return __ret;
}

/* --- Get high/low halves --- */

static __inline__ uint8x8_t __attribute__((__always_inline__))
vget_low_u8(uint8x16_t __a)
{
    uint8x8_t __ret;
    __builtin_memcpy(&__ret.__val[0], &__a.__val[0], 8);
    return __ret;
}

static __inline__ uint8x8_t __attribute__((__always_inline__))
vget_high_u8(uint8x16_t __a)
{
    uint8x8_t __ret;
    __builtin_memcpy(&__ret.__val[0], &__a.__val[8], 8);
    return __ret;
}

static __inline__ int8x8_t __attribute__((__always_inline__))
vget_low_s8(int8x16_t __a)
{
    int8x8_t __ret;
    __builtin_memcpy(&__ret.__val[0], &__a.__val[0], 8);
    return __ret;
}

static __inline__ int8x8_t __attribute__((__always_inline__))
vget_high_s8(int8x16_t __a)
{
    int8x8_t __ret;
    __builtin_memcpy(&__ret.__val[0], &__a.__val[8], 8);
    return __ret;
}

static __inline__ uint64x1_t __attribute__((__always_inline__))
vget_low_u64(uint64x2_t __a)
{
    uint64x1_t __ret;
    __ret.__val[0] = __a.__val[0];
    return __ret;
}

static __inline__ uint64x1_t __attribute__((__always_inline__))
vget_high_u64(uint64x2_t __a)
{
    uint64x1_t __ret;
    __ret.__val[0] = __a.__val[1];
    return __ret;
}

static __inline__ poly64x1_t __attribute__((__always_inline__))
vget_low_p64(poly64x2_t __a)
{
    poly64x1_t __ret;
    __ret.__val[0] = __a.__val[0];
    return __ret;
}

/* ================================================================== */
/*                       LANE ACCESS                                   */
/* ================================================================== */

static __inline__ signed char __attribute__((__always_inline__))
vget_lane_s8(int8x8_t __a, int __lane)
{
    return __a.__val[__lane];
}

static __inline__ unsigned char __attribute__((__always_inline__))
vget_lane_u8(uint8x8_t __a, int __lane)
{
    return __a.__val[__lane];
}

static __inline__ unsigned int __attribute__((__always_inline__))
vgetq_lane_u32(uint32x4_t __a, int __lane)
{
    return __a.__val[__lane];
}

/* ================================================================== */
/*                    ARITHMETIC OPERATIONS                            */
/* ================================================================== */

static __inline__ uint8x16_t __attribute__((__always_inline__))
vaddq_u8(uint8x16_t __a, uint8x16_t __b)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = __a.__val[__i] + __b.__val[__i];
    return __ret;
}

static __inline__ uint8x8_t __attribute__((__always_inline__))
vadd_u8(uint8x8_t __a, uint8x8_t __b)
{
    uint8x8_t __ret;
    for (int __i = 0; __i < 8; __i++)
        __ret.__val[__i] = __a.__val[__i] + __b.__val[__i];
    return __ret;
}

/* --- Widening arithmetic --- */

static __inline__ uint16x8_t __attribute__((__always_inline__))
vaddl_u8(uint8x8_t __a, uint8x8_t __b)
{
    /* Widening add: u8 + u8 -> u16 */
    uint16x8_t __ret;
    for (int __i = 0; __i < 8; __i++)
        __ret.__val[__i] = (unsigned short)__a.__val[__i] + (unsigned short)__b.__val[__i];
    return __ret;
}

static __inline__ uint16x8_t __attribute__((__always_inline__))
vabdl_u8(uint8x8_t __a, uint8x8_t __b)
{
    /* Widening absolute difference: |a - b| -> u16 */
    uint16x8_t __ret;
    for (int __i = 0; __i < 8; __i++) {
        int __d = (int)__a.__val[__i] - (int)__b.__val[__i];
        __ret.__val[__i] = (unsigned short)(__d < 0 ? -__d : __d);
    }
    return __ret;
}

static __inline__ uint16x8_t __attribute__((__always_inline__))
vabdq_u16(uint16x8_t __a, uint16x8_t __b)
{
    uint16x8_t __ret;
    for (int __i = 0; __i < 8; __i++) {
        int __d = (int)__a.__val[__i] - (int)__b.__val[__i];
        __ret.__val[__i] = (unsigned short)(__d < 0 ? -__d : __d);
    }
    return __ret;
}

static __inline__ uint8x8_t __attribute__((__always_inline__))
vhadd_u8(uint8x8_t __a, uint8x8_t __b)
{
    /* Halving add: (a + b) >> 1 without overflow */
    uint8x8_t __ret;
    for (int __i = 0; __i < 8; __i++)
        __ret.__val[__i] = ((unsigned short)__a.__val[__i] + (unsigned short)__b.__val[__i]) >> 1;
    return __ret;
}

/* --- Narrowing --- */

static __inline__ uint8x8_t __attribute__((__always_inline__))
vmovn_u16(uint16x8_t __a)
{
    /* Narrow: take low 8 bits of each u16 */
    uint8x8_t __ret;
    for (int __i = 0; __i < 8; __i++)
        __ret.__val[__i] = (unsigned char)__a.__val[__i];
    return __ret;
}

/* ================================================================== */
/*                    COMPARISON OPERATIONS                            */
/* ================================================================== */

static __inline__ uint8x16_t __attribute__((__always_inline__))
vceqq_u8(uint8x16_t __a, uint8x16_t __b)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++)
        __ret.__val[__i] = (__a.__val[__i] == __b.__val[__i]) ? 0xFF : 0x00;
    return __ret;
}

static __inline__ uint16x8_t __attribute__((__always_inline__))
vcleq_u16(uint16x8_t __a, uint16x8_t __b)
{
    /* Compare less-than-or-equal unsigned */
    uint16x8_t __ret;
    for (int __i = 0; __i < 8; __i++)
        __ret.__val[__i] = (__a.__val[__i] <= __b.__val[__i]) ? 0xFFFF : 0x0000;
    return __ret;
}

/* ================================================================== */
/*                    EXTRACT / SHUFFLE                                */
/* ================================================================== */

static __inline__ uint8x16_t __attribute__((__always_inline__))
vextq_u8(uint8x16_t __a, uint8x16_t __b, int __n)
{
    /* Extract: concatenate a and b, then extract 16 bytes starting at byte __n */
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++) {
        int __idx = __i + __n;
        __ret.__val[__i] = (__idx < 16) ? __a.__val[__idx] : __b.__val[__idx - 16];
    }
    return __ret;
}

static __inline__ uint8x8_t __attribute__((__always_inline__))
vext_u8(uint8x8_t __a, uint8x8_t __b, int __n)
{
    uint8x8_t __ret;
    for (int __i = 0; __i < 8; __i++) {
        int __idx = __i + __n;
        __ret.__val[__i] = (__idx < 8) ? __a.__val[__idx] : __b.__val[__idx - 8];
    }
    return __ret;
}

/* ================================================================== */
/*                 POLYNOMIAL MULTIPLICATION                           */
/* ================================================================== */

static __inline__ poly8x16_t __attribute__((__always_inline__))
vmulq_p8(poly8x16_t __a, poly8x16_t __b)
{
    /* Polynomial (carry-less) multiply per byte */
    poly8x16_t __ret;
    for (int __i = 0; __i < 16; __i++) {
        unsigned char __r = 0;
        unsigned char __av = __a.__val[__i];
        unsigned char __bv = __b.__val[__i];
        for (int __j = 0; __j < 8; __j++) {
            if (__bv & (1 << __j))
                __r ^= __av << __j;
        }
        __ret.__val[__i] = __r;
    }
    return __ret;
}

/* 64-bit polynomial multiply (for crypto/GHASH).
 * Scalar C implementation of carry-less multiplication - functionally correct
 * but much slower than the hardware PMULL instruction. */
static __inline__ poly128_t __attribute__((__always_inline__))
vmull_p64(poly64_t __a, poly64_t __b)
{
    /* Carry-less multiply of two 64-bit polynomials -> 128-bit result */
    __uint128_t __r = 0;
    for (int __j = 0; __j < 64; __j++) {
        if (__b & (1ULL << __j))
            __r ^= (__uint128_t)__a << __j;
    }
    return (poly128_t)__r;
}

static __inline__ poly128_t __attribute__((__always_inline__))
vmull_high_p64(poly64x2_t __a, poly64x2_t __b)
{
    return vmull_p64(__a.__val[1], __b.__val[1]);
}

/* ================================================================== */
/*                    REVERSE OPERATIONS                               */
/* ================================================================== */

static __inline__ uint16x8_t __attribute__((__always_inline__))
vrev32q_u16(uint16x8_t __a)
{
    /* Reverse 16-bit elements within each 32-bit word */
    uint16x8_t __ret;
    for (int __i = 0; __i < 8; __i += 2) {
        __ret.__val[__i] = __a.__val[__i + 1];
        __ret.__val[__i + 1] = __a.__val[__i];
    }
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vrbitq_u8(uint8x16_t __a)
{
    /* Reverse bits within each byte */
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++) {
        unsigned char __v = __a.__val[__i];
        unsigned char __r = 0;
        for (int __j = 0; __j < 8; __j++)
            __r |= ((__v >> __j) & 1) << (7 - __j);
        __ret.__val[__i] = __r;
    }
    return __ret;
}

/* ================================================================== */
/*                   TABLE LOOKUP OPERATIONS                           */
/* ================================================================== */

static __inline__ uint8x16_t __attribute__((__always_inline__))
vqtbl1q_u8(uint8x16_t __a, uint8x16_t __b)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++) {
        unsigned char __idx = __b.__val[__i];
        __ret.__val[__i] = (__idx < 16) ? __a.__val[__idx] : 0;
    }
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vqtbx1q_u8(uint8x16_t __def, uint8x16_t __a, uint8x16_t __b)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++) {
        unsigned char __idx = __b.__val[__i];
        __ret.__val[__i] = (__idx < 16) ? __a.__val[__idx] : __def.__val[__i];
    }
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vqtbl4q_u8(uint8x16x4_t __a, uint8x16_t __b)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++) {
        unsigned char __idx = __b.__val[__i];
        if (__idx < 16)
            __ret.__val[__i] = __a.val[0].__val[__idx];
        else if (__idx < 32)
            __ret.__val[__i] = __a.val[1].__val[__idx - 16];
        else if (__idx < 48)
            __ret.__val[__i] = __a.val[2].__val[__idx - 32];
        else if (__idx < 64)
            __ret.__val[__i] = __a.val[3].__val[__idx - 48];
        else
            __ret.__val[__i] = 0;
    }
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vqtbx4q_u8(uint8x16_t __def, uint8x16x4_t __a, uint8x16_t __b)
{
    uint8x16_t __ret;
    for (int __i = 0; __i < 16; __i++) {
        unsigned char __idx = __b.__val[__i];
        if (__idx < 16)
            __ret.__val[__i] = __a.val[0].__val[__idx];
        else if (__idx < 32)
            __ret.__val[__i] = __a.val[1].__val[__idx - 16];
        else if (__idx < 48)
            __ret.__val[__i] = __a.val[2].__val[__idx - 32];
        else if (__idx < 64)
            __ret.__val[__i] = __a.val[3].__val[__idx - 48];
        else
            __ret.__val[__i] = __def.__val[__i];
    }
    return __ret;
}

/* 64-bit table lookup (AArch32 compat) */

static __inline__ uint8x8_t __attribute__((__always_inline__))
vtbl2_u8(uint8x8x2_t __a, uint8x8_t __b)
{
    uint8x8_t __ret;
    for (int __i = 0; __i < 8; __i++) {
        unsigned char __idx = __b.__val[__i];
        if (__idx < 8)
            __ret.__val[__i] = __a.val[0].__val[__idx];
        else if (__idx < 16)
            __ret.__val[__i] = __a.val[1].__val[__idx - 8];
        else
            __ret.__val[__i] = 0;
    }
    return __ret;
}

static __inline__ uint8x8_t __attribute__((__always_inline__))
vtbx2_u8(uint8x8_t __def, uint8x8x2_t __a, uint8x8_t __b)
{
    uint8x8_t __ret;
    for (int __i = 0; __i < 8; __i++) {
        unsigned char __idx = __b.__val[__i];
        if (__idx < 8)
            __ret.__val[__i] = __a.val[0].__val[__idx];
        else if (__idx < 16)
            __ret.__val[__i] = __a.val[1].__val[__idx - 8];
        else
            __ret.__val[__i] = __def.__val[__i];
    }
    return __ret;
}

/* ================================================================== */
/*                   PAIRWISE / ACROSS-LANE                            */
/* ================================================================== */

static __inline__ int8x8_t __attribute__((__always_inline__))
vpmin_s8(int8x8_t __a, int8x8_t __b)
{
    int8x8_t __ret;
    for (int __i = 0; __i < 4; __i++) {
        signed char __x = __a.__val[2 * __i];
        signed char __y = __a.__val[2 * __i + 1];
        __ret.__val[__i] = (__x < __y) ? __x : __y;
    }
    for (int __i = 0; __i < 4; __i++) {
        signed char __x = __b.__val[2 * __i];
        signed char __y = __b.__val[2 * __i + 1];
        __ret.__val[4 + __i] = (__x < __y) ? __x : __y;
    }
    return __ret;
}

static __inline__ signed char __attribute__((__always_inline__))
vminvq_s8(int8x16_t __a)
{
    signed char __min = __a.__val[0];
    for (int __i = 1; __i < 16; __i++) {
        if (__a.__val[__i] < __min)
            __min = __a.__val[__i];
    }
    return __min;
}

/* ================================================================== */
/*                       AES INTRINSICS                                */
/* ================================================================== */
/* These are stubs for parsing. The kernel uses inline asm for AES. */

static __inline__ uint8x16_t __attribute__((__always_inline__))
vaeseq_u8(uint8x16_t __a, uint8x16_t __b)
{
    (void)__b;
    return __a;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vaesmcq_u8(uint8x16_t __a)
{
    return __a;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vaesdq_u8(uint8x16_t __a, uint8x16_t __b)
{
    (void)__b;
    return __a;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vaesimcq_u8(uint8x16_t __a)
{
    return __a;
}

/* ================================================================== */
/*                  TYPE REINTERPRET CASTS                              */
/* ================================================================== */

static __inline__ int8x16_t __attribute__((__always_inline__))
vreinterpretq_s8_u8(uint8x16_t __a)
{
    int8x16_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vreinterpretq_u8_s8(int8x16_t __a)
{
    uint8x16_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vreinterpretq_u8_u16(uint16x8_t __a)
{
    uint8x16_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint16x8_t __attribute__((__always_inline__))
vreinterpretq_u16_u8(uint8x16_t __a)
{
    uint16x8_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vreinterpretq_u8_u32(uint32x4_t __a)
{
    uint8x16_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint32x4_t __attribute__((__always_inline__))
vreinterpretq_u32_u8(uint8x16_t __a)
{
    uint32x4_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vreinterpretq_u8_u64(uint64x2_t __a)
{
    uint8x16_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint64x2_t __attribute__((__always_inline__))
vreinterpretq_u64_u8(uint8x16_t __a)
{
    uint64x2_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vreinterpretq_u8_p8(poly8x16_t __a)
{
    uint8x16_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ poly8x16_t __attribute__((__always_inline__))
vreinterpretq_p8_u8(uint8x16_t __a)
{
    poly8x16_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ poly64x2_t __attribute__((__always_inline__))
vreinterpretq_p64_u8(uint8x16_t __a)
{
    poly64x2_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint64x2_t __attribute__((__always_inline__))
vreinterpretq_u64_p64(poly64x2_t __a)
{
    uint64x2_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

static __inline__ uint8x16_t __attribute__((__always_inline__))
vreinterpretq_u8_p128(poly128_t __a)
{
    uint8x16_t __ret;
    __builtin_memcpy(&__ret, &__a, 16);
    return __ret;
}

#endif /* _ARM_NEON_H_INCLUDED */
