/* CCC compiler bundled mmintrin.h - MMX intrinsics */
#ifndef _MMINTRIN_H_INCLUDED
#define _MMINTRIN_H_INCLUDED

/* MMX intrinsics are only available on x86/x86-64 targets */
#if !defined(__x86_64__) && !defined(__i386__) && !defined(__i686__)
#error "MMX intrinsics (mmintrin.h) require an x86 target"
#endif

/* __m64: 64-bit MMX vector type.
 * Used by SSE storel/storeh intrinsics and legacy MMX code. */
typedef struct __attribute__((__aligned__(8))) {
    long long __val;
} __m64;

#endif /* _MMINTRIN_H_INCLUDED */
