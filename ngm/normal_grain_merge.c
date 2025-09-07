#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <smmintrin.h>
#include <immintrin.h>  /* AVX2 + SSE4.2 */

/* ----- Runtime CPU feature detection (GCC/Clang + MSVC) ----- */
#if defined(_MSC_VER)
  #include <intrin.h>
  static int os_supports_avx(void) {
      /* Check OSXSAVE + XCR0[2:1] == 11b so OS saves YMM state */
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      int ecx = cpuInfo[2];
      int osxsave = (ecx >> 27) & 1;
      if (!osxsave) return 0;
      unsigned long long xcr0 = _xgetbv(0);
      return ((xcr0 & 0x6) == 0x6); /* XMM (bit1) and YMM (bit2) state enabled */
  }

  static int cpu_supports_avx2(void) {
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      int ecx = cpuInfo[2];
      int avx   = (ecx >> 28) & 1;
      int osxsave = (ecx >> 27) & 1;
      if (!(avx && osxsave && os_supports_avx())) return 0;

      /* Leaf 7, subleaf 0: EBX bit 5 = AVX2 */
      int ex[4];
      __cpuidex(ex, 7, 0);
      int ebx = ex[1];
      return (ebx >> 5) & 1;
  }

  static int cpu_supports_sse42(void) {
      int cpuInfo[4];
      __cpuid(cpuInfo, 1);
      int ecx = cpuInfo[2];
      return (ecx >> 20) & 1; /* SSE4.2 */
  }
#else
  /* GCC/Clang path */
  static int os_supports_avx(void) {
  #if defined(__GNUC__) || defined(__clang__)
      /* If we’re here, assume OS supports AVX when the CPU supports it.
         For full rigor you can also call xgetbv via inline asm, but it’s uncommon to lack it. */
      return 1;
  #else
      return 0;
  #endif
  }

  static int cpu_supports_avx2(void) {
  #if defined(__GNUC__) || defined(__clang__)
      /* Requires -mavx2 at compile, but we only *call* the AVX2 kernel if true. */
      return __builtin_cpu_supports("avx2");
  #else
      return 0;
  #endif
  }

  static int cpu_supports_sse42(void) {
  #if defined(__GNUC__) || defined(__clang__)
      return __builtin_cpu_supports("sse4.2");
  #else
      return 0;
  #endif
  }
#endif

#define SKIN_WEIGHT 0.3f

typedef enum {
    KERNEL_AUTO = 0,
    KERNEL_SCALAR = 1,
    KERNEL_SSE42 = 2,
    KERNEL_AVX2 = 3
} kernel_kind;

/* ---------- Utility: safe views, shape checks ---------- */

/* Make a new uint8, C-contiguous, aligned view we own. Never DECREF the input obj. */
static inline int get_uint8_c_contig(PyObject *obj, PyArrayObject **out, const char *name) {
    const int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;
    PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_UINT8, flags);
    if (!arr) {
        PyErr_Format(PyExc_TypeError, "%s must be a uint8 ndarray", name);
        return 0;
    }
    *out = arr;  /* new reference */
    return 1;
}

static inline int ensure_uint8_contig(PyArrayObject **arr, const char *name) {
    PyArrayObject *tmp = (PyArrayObject*)PyArray_FROM_OTF(
        (PyObject*)(*arr), NPY_UINT8, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!tmp) return 0;
    Py_XDECREF(*arr);
    *arr = tmp;
    return 1;
}

static inline int check_shape_requirements(PyArrayObject *base,
                                    PyArrayObject *texture,
                                    PyArrayObject *skin,
                                    PyArrayObject *im_alpha,
                                    int *texture_has_alpha,
                                    npy_intp *height,
                                    npy_intp *width) {
    if (PyArray_NDIM(base) != 3 || PyArray_DIMS(base)[2] != 3) {
        PyErr_SetString(PyExc_ValueError, "base must have shape (H, W, 3)");
        return 0;
    }
    if (PyArray_NDIM(texture) != 3) {
        PyErr_SetString(PyExc_ValueError, "texture must have shape (H, W, 3) or (H, W, 4)");
        return 0;
    }
    npy_intp tc = PyArray_DIMS(texture)[2];
    if (!(tc == 3 || tc == 4)) {
        PyErr_SetString(PyExc_ValueError, "texture must have 3 or 4 channels");
        return 0;
    }
    *texture_has_alpha = (tc == 4);

    if (PyArray_NDIM(skin) != 3 || PyArray_DIMS(skin)[2] != 4) {
        PyErr_SetString(PyExc_ValueError, "skin must have shape (H, W, 4)");
        return 0;
    }
    if (PyArray_NDIM(im_alpha) != 2) {
        PyErr_SetString(PyExc_ValueError, "im_alpha must have shape (H, W)");
        return 0;
    }

    npy_intp h = PyArray_DIMS(base)[0], w = PyArray_DIMS(base)[1];
    if (PyArray_DIMS(texture)[0] != h || PyArray_DIMS(texture)[1] != w ||
        PyArray_DIMS(skin)[0] != h    || PyArray_DIMS(skin)[1] != w ||
        PyArray_DIMS(im_alpha)[0] != h|| PyArray_DIMS(im_alpha)[1] != w) {
        PyErr_SetString(PyExc_ValueError, "All inputs must share the same H and W");
        return 0;
    }
    *height = h; *width = w;
    return 1;
}

/* ---------- Scalar reference kernel (clear, correct, easy to modify) ---------- */
/* Converts uint8 to float32 in [0,1], does placeholder math, writes back to uint8. */
/* Replace the placeholder math with your blend. */

/*
 * Converts nan and inf values to 0 and 255 respectively.
 */
static inline float nan_to_num(float x) {
    if (isnan(x)) {
        return 0.0f;  // replace NaN with 0
    } 
    if (isinf(x)) {
        if (x > 0) {
            return 255.0f;  // positive infinity -> max uint8
        } else {
            return 0.0f; // negative infinity -> min uint8
        }
    } 
    else {
        return x; // keep finite values as they are
    }
}

/*
 * Scaler kernel for RGB texture input.
 */
static void kernel_scalar_rgb(const uint8_t *base, const uint8_t *texture,
                              const uint8_t *skin, const uint8_t *im_alpha,
                              uint8_t *out, npy_intp pixels) {
    for (npy_intp i = 0; i < pixels; ++i) {
        const uint8_t b_r = base[3*i+0];
        const uint8_t b_g = base[3*i+1];
        const uint8_t b_b = base[3*i+2];

        const uint8_t t_r = texture[3*i+0];
        const uint8_t t_g = texture[3*i+1];
        const uint8_t t_b = texture[3*i+2];

        const uint8_t s_r = skin[4*i+0];
        const uint8_t s_g = skin[4*i+1];
        const uint8_t s_b = skin[4*i+2];
        const uint8_t s_a = skin[4*i+3];

        const uint8_t a_im = im_alpha[i];

        /* float32 intermediates in [0,1] */
        const float fb_r = b_r * (1.0f/255.0f);
        const float fb_g = b_g * (1.0f/255.0f);
        const float fb_b = b_b * (1.0f/255.0f);

        const float ft_r = t_r * (1.0f/255.0f);
        const float ft_g = t_g * (1.0f/255.0f);
        const float ft_b = t_b * (1.0f/255.0f);

        const float fs_r = s_r * (1.0f/255.0f);
        const float fs_g = s_g * (1.0f/255.0f);
        const float fs_b = s_b * (1.0f/255.0f);
        const float fs_a = s_a * (1.0f/255.0f);

        const float fa_im = a_im * (1.0f/255.0f);

        /*
         **********************
         * normal grain merge *
         **********************
         */

        /* inverse_tpa */
        float fit_a = 1.0f - fa_im;
        /* gm_out = np.clip(texture + skin - 0.5, 0.0, 1.0) */
        float fr = ft_r + fs_r - 0.5f;
        float fg = ft_g + fs_g - 0.5f;
        float fb = ft_b + fs_b - 0.5f;
        /* np.clip */
        fr = fr < 0.0f ? 0.0f : (fr > 1.0f ? 1.0f : fr);
        fg = fg < 0.0f ? 0.0f : (fg > 1.0f ? 1.0f : fg);
        fb = fb < 0.0f ? 0.0f : (fb > 1.0f ? 1.0f : fb);
        /* gm_out = gm_out * texture_alpha + texture * inverse_tpa */
        fr = fr * fa_im + ft_r * fit_a;
        fg = fg * fa_im + ft_g * fit_a;
        fb = fb * fa_im + ft_b * fit_a;

        /* gm_out = gm_out * (1 - SKIN_WEIGHT) + (skin * SKIN_WEIGHT) */
        fr = fr * (1.0f - SKIN_WEIGHT) + fs_r * SKIN_WEIGHT;
        fg = fg * (1.0f - SKIN_WEIGHT) + fs_g * SKIN_WEIGHT;
        fb = fb * (1.0f - SKIN_WEIGHT) + fs_b * SKIN_WEIGHT;

        /* np.nan_to_num(gm_out, copy=False) */
        fr = nan_to_num(fr);
        fg = nan_to_num(fg);
        fb = nan_to_num(fb);

        /* Normal merge
         * n_out = gm_out * texture_alpha + base * inverse_tpa
         * 
         * In this case, texture_alpha is the skin alpha since texture doesn't have an alpha channel here.
         */
        fr = fr * fa_im + fb_r * fit_a;
        fg = fg * fa_im + fb_g * fit_a;
        fb = fb * fa_im + fb_b * fit_a;


        out[3*i+0] = (uint8_t)(fr * 255.0f);
        out[3*i+1] = (uint8_t)(fg * 255.0f);
        out[3*i+2] = (uint8_t)(fb * 255.0f);
    }
}

static void kernel_scalar_rgba(const uint8_t *base, const uint8_t *texture,
                               const uint8_t *skin, const uint8_t *im_alpha,
                               uint8_t *out, npy_intp pixels) {
    for (npy_intp i = 0; i < pixels; ++i) {
        const uint8_t b_r = base[3*i+0];
        const uint8_t b_g = base[3*i+1];
        const uint8_t b_b = base[3*i+2];

        const uint8_t t_r = texture[4*i+0];
        const uint8_t t_g = texture[4*i+1];
        const uint8_t t_b = texture[4*i+2];
        const uint8_t t_a = texture[4*i+3];  /* present in RGBA branch */

        const uint8_t s_r = skin[4*i+0];
        const uint8_t s_g = skin[4*i+1];
        const uint8_t s_b = skin[4*i+2];
        const uint8_t s_a = skin[4*i+3];

        const uint8_t a_im = im_alpha[i];

        const float fb_r = b_r * (1.0f/255.0f);
        const float fb_g = b_g * (1.0f/255.0f);
        const float fb_b = b_b * (1.0f/255.0f);

        const float ft_r = t_r * (1.0f/255.0f);
        const float ft_g = t_g * (1.0f/255.0f);
        const float ft_b = t_b * (1.0f/255.0f);
        float ft_a = t_a * (1.0f/255.0f);

        const float fs_r = s_r * (1.0f/255.0f);
        const float fs_g = s_g * (1.0f/255.0f);
        const float fs_b = s_b * (1.0f/255.0f);
        const float fs_a = s_a * (1.0f/255.0f);

        const float fa_im = a_im * (1.0f/255.0f);

        /*
         **********************
         * normal grain merge *
         **********************
         */
        /* Merge texture and skin alphas */

        /* texture_alpha = texture[..., 3] * im_alpha*/
        ft_a = ft_a * fa_im;
        /* inverse_tpa = 1 - texture_alpha */
        float fit_a = 1.0f - ft_a;

        /* gm_out = np.clip(texture + skin - 0.5, 0.0, 1.0) */
        float fr = ft_r + fs_r - 0.5f;
        float fg = ft_g + fs_g - 0.5f;
        float fb = ft_b + fs_b - 0.5f;
        /* np.clip */
        fr = fr < 0.0f ? 0.0f : (fr > 1.0f ? 1.0f : fr);
        fg = fg < 0.0f ? 0.0f : (fg > 1.0f ? 1.0f : fg);
        fb = fb < 0.0f ? 0.0f : (fb > 1.0f ? 1.0f : fb);

        /* gm_out = gm_out * texture_alpha + texture * inverse_tpa */
        fr = fr * ft_a + ft_r * fit_a;
        fg = fg * ft_a + ft_g * fit_a;
        fb = fb * ft_a + ft_b * fit_a;


        /* gm_out = gm_out * (1 - SKIN_WEIGHT) + (skin * SKIN_WEIGHT) */
        fr = fr * (1.0f - SKIN_WEIGHT) + fs_r * SKIN_WEIGHT;
        fg = fg * (1.0f - SKIN_WEIGHT) + fs_g * SKIN_WEIGHT;
        fb = fb * (1.0f - SKIN_WEIGHT) + fs_b * SKIN_WEIGHT;

        /* np.nan_to_num(gm_out, copy=False) */
        fr = nan_to_num(fr);
        fg = nan_to_num(fg);
        fb = nan_to_num(fb);

        /* Normal merge
         * n_out = gm_out * texture_alpha + base * inverse_tpa
         */
        fr = fr * ft_a + fb_r * fit_a;
        fg = fg * ft_a + fb_g * fit_a;
        fb = fb * ft_a + fb_b * fit_a;

        out[3*i+0] = (uint8_t)(fr * 255.0f);
        out[3*i+1] = (uint8_t)(fg * 255.0f);
        out[3*i+2] = (uint8_t)(fb * 255.0f);
    }
}

/* ---------- AVX2 helpers ----------
   Interleaved RGB(A) is awkward for SIMD. For a skeleton that still uses AVX2, we use gathers
   over a stride (3 or 4) to pull 8 pixels for a given channel into a vector.
   You can later replace gathers with better deinterleaving if needed.
*/

/* Convert 8 u8 interleaved channel samples (stride 3 or 4) to float32 in [0,1] via gather. */
static inline __m256 gather_u8_to_unit_f32_avx2(const uint8_t *base_ptr, int stride,
                                                npy_intp start_idx) {
    const int i0 = (int)((start_idx + 0) * stride);
    const int i1 = (int)((start_idx + 1) * stride);
    const int i2 = (int)((start_idx + 2) * stride);
    const int i3 = (int)((start_idx + 3) * stride);
    const int i4 = (int)((start_idx + 4) * stride);
    const int i5 = (int)((start_idx + 5) * stride);
    const int i6 = (int)((start_idx + 6) * stride);
    const int i7 = (int)((start_idx + 7) * stride);

    __m256i offs = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
    __m256i v32  = _mm256_i32gather_epi32((const int*)base_ptr, offs, 1); /* read 8 x u8 as u32 */
    v32 = _mm256_and_si256(v32, _mm256_set1_epi32(0xFF));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(v32), _mm256_set1_ps(1.0f/255.0f));
}

/* Convert 8 consecutive u8 to float32 in [0,1] (for grayscale im_alpha). */
static inline __m256 load8_u8_to_unit_f32_avx2(const uint8_t *p) {
    __m128i v8  = _mm_loadl_epi64((const __m128i*)p);        /* 8 bytes -> XMM */
    __m256i v32 = _mm256_cvtepu8_epi32(v8);                  /* widen to 8 x u32 */
    return _mm256_mul_ps(_mm256_cvtepi32_ps(v32), _mm256_set1_ps(1.0f/255.0f));
}

static inline __m256 clamp01_ps(__m256 x) {
    return _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
}

/* Replace NaN with 0.0f (Inf is not expected from uint8-origin math). */
static inline __m256 nan_to_num_ps(__m256 x) {
    __m256 cmp = _mm256_cmp_ps(x, x, _CMP_ORD_Q); /* 0 for NaN lanes */
    return _mm256_blendv_ps(_mm256_set1_ps(0.0f), x, cmp);
}

/* Truncate [0,1] floats to uint8 and scatter to interleaved RGB output. */
static inline void store_unit_f32_to_u8_rgb8_avx2(__m256 fr, __m256 fg, __m256 fb,
                                                  uint8_t *out_ptr, npy_intp start_idx) {
    __m256 scale = _mm256_set1_ps(255.0f);
    __m256i ir = _mm256_cvttps_epi32(_mm256_mul_ps(fr, scale));
    __m256i ig = _mm256_cvttps_epi32(_mm256_mul_ps(fg, scale));
    __m256i ib = _mm256_cvttps_epi32(_mm256_mul_ps(fb, scale));

    int r[8], g[8], b[8];
    _mm256_storeu_si256((__m256i*)r, ir);
    _mm256_storeu_si256((__m256i*)g, ig);
    _mm256_storeu_si256((__m256i*)b, ib);

    for (int k = 0; k < 8; ++k) {
        const npy_intp p = start_idx + k;
        out_ptr[3*p+0] = (uint8_t)(r[k] < 0 ? 0 : r[k] > 255 ? 255 : r[k]);
        out_ptr[3*p+1] = (uint8_t)(g[k] < 0 ? 0 : g[k] > 255 ? 255 : g[k]);
        out_ptr[3*p+2] = (uint8_t)(b[k] < 0 ? 0 : b[k] > 255 ? 255 : b[k]);
    }
}

/* texture is RGB: texture_alpha = im_alpha broadcast, inverse_tpa = 1 - texture_alpha */
static void kernel_avx2_rgb(const uint8_t *base, const uint8_t *texture,
                            const uint8_t *skin, const uint8_t *im_alpha,
                            uint8_t *out, npy_intp pixels) {
    const int stride3 = 3, stride4 = 4;
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 w    = _mm256_set1_ps((float)SKIN_WEIGHT);
    const __m256 invw = _mm256_set1_ps(1.0f - (float)SKIN_WEIGHT);

    npy_intp i = 0;
    for (; i + 8 <= pixels; i += 8) {
        /* base RGB in [0,1] */
        __m256 fb_r = gather_u8_to_unit_f32_avx2(base+0, stride3, i);
        __m256 fb_g = gather_u8_to_unit_f32_avx2(base+1, stride3, i);
        __m256 fb_b = gather_u8_to_unit_f32_avx2(base+2, stride3, i);

        /* texture RGB in [0,1] */
        __m256 ft_r = gather_u8_to_unit_f32_avx2(texture+0, stride3, i);
        __m256 ft_g = gather_u8_to_unit_f32_avx2(texture+1, stride3, i);
        __m256 ft_b = gather_u8_to_unit_f32_avx2(texture+2, stride3, i);

        /* skin RGB in [0,1] */
        __m256 fs_r = gather_u8_to_unit_f32_avx2(skin+0, stride4, i);
        __m256 fs_g = gather_u8_to_unit_f32_avx2(skin+1, stride4, i);
        __m256 fs_b = gather_u8_to_unit_f32_avx2(skin+2, stride4, i);

        /* texture_alpha = im_alpha */
        __m256 fa_im = load8_u8_to_unit_f32_avx2(im_alpha + i);
        __m256 fit_a = _mm256_sub_ps(one, fa_im);

        /* gm_out = clip(texture + skin - 0.5) */
        __m256 gm_r = clamp01_ps(_mm256_sub_ps(_mm256_add_ps(ft_r, fs_r), half));
        __m256 gm_g = clamp01_ps(_mm256_sub_ps(_mm256_add_ps(ft_g, fs_g), half));
        __m256 gm_b = clamp01_ps(_mm256_sub_ps(_mm256_add_ps(ft_b, fs_b), half));

        /* gm_out = gm_out * texture_alpha + texture * inverse_tpa */
        gm_r = _mm256_add_ps(_mm256_mul_ps(gm_r, fa_im), _mm256_mul_ps(ft_r, fit_a));
        gm_g = _mm256_add_ps(_mm256_mul_ps(gm_g, fa_im), _mm256_mul_ps(ft_g, fit_a));
        gm_b = _mm256_add_ps(_mm256_mul_ps(gm_b, fa_im), _mm256_mul_ps(ft_b, fit_a));

        /* gm_out = gm_out * (1 - w) + skin * w */
        gm_r = _mm256_add_ps(_mm256_mul_ps(gm_r, invw), _mm256_mul_ps(fs_r, w));
        gm_g = _mm256_add_ps(_mm256_mul_ps(gm_g, invw), _mm256_mul_ps(fs_g, w));
        gm_b = _mm256_add_ps(_mm256_mul_ps(gm_b, invw), _mm256_mul_ps(fs_b, w));

        /* nan_to_num */
        gm_r = nan_to_num_ps(gm_r);
        gm_g = nan_to_num_ps(gm_g);
        gm_b = nan_to_num_ps(gm_b);

        /* n_out = gm_out * texture_alpha + base * inverse_tpa */
        __m256 fr = _mm256_add_ps(_mm256_mul_ps(gm_r, fa_im), _mm256_mul_ps(fb_r, fit_a));
        __m256 fg = _mm256_add_ps(_mm256_mul_ps(gm_g, fa_im), _mm256_mul_ps(fb_g, fit_a));
        __m256 fb = _mm256_add_ps(_mm256_mul_ps(gm_b, fa_im), _mm256_mul_ps(fb_b, fit_a));

        store_unit_f32_to_u8_rgb8_avx2(fr, fg, fb, out, i);
    }

    if (i < pixels) {
        kernel_scalar_rgb(base + 3*i, texture + 3*i, skin + 4*i, im_alpha + i,
                          out + 3*i, pixels - i);
    }
}

/* texture is RGBA: texture_alpha = texture.A * im_alpha, inverse_tpa = 1 - texture_alpha */
static void kernel_avx2_rgba(const uint8_t *base, const uint8_t *texture,
                             const uint8_t *skin, const uint8_t *im_alpha,
                             uint8_t *out, npy_intp pixels) {
    const int stride3 = 3, stride4 = 4;
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 w    = _mm256_set1_ps((float)SKIN_WEIGHT);
    const __m256 invw = _mm256_set1_ps(1.0f - (float)SKIN_WEIGHT);

    npy_intp i = 0;
    for (; i + 8 <= pixels; i += 8) {
        __m256 fb_r = gather_u8_to_unit_f32_avx2(base+0, stride3, i);
        __m256 fb_g = gather_u8_to_unit_f32_avx2(base+1, stride3, i);
        __m256 fb_b = gather_u8_to_unit_f32_avx2(base+2, stride3, i);

        __m256 ft_r = gather_u8_to_unit_f32_avx2(texture+0, stride4, i);
        __m256 ft_g = gather_u8_to_unit_f32_avx2(texture+1, stride4, i);
        __m256 ft_b = gather_u8_to_unit_f32_avx2(texture+2, stride4, i);
        __m256 ft_a = gather_u8_to_unit_f32_avx2(texture+3, stride4, i);  /* texture alpha */

        __m256 fs_r = gather_u8_to_unit_f32_avx2(skin+0, stride4, i);
        __m256 fs_g = gather_u8_to_unit_f32_avx2(skin+1, stride4, i);
        __m256 fs_b = gather_u8_to_unit_f32_avx2(skin+2, stride4, i);

        __m256 fa_im = load8_u8_to_unit_f32_avx2(im_alpha + i);
        __m256 fta   = _mm256_mul_ps(ft_a, fa_im);           /* texture_alpha */
        __m256 fit_a = _mm256_sub_ps(one, fta);               /* inverse_tpa  */

        __m256 gm_r = clamp01_ps(_mm256_sub_ps(_mm256_add_ps(ft_r, fs_r), half));
        __m256 gm_g = clamp01_ps(_mm256_sub_ps(_mm256_add_ps(ft_g, fs_g), half));
        __m256 gm_b = clamp01_ps(_mm256_sub_ps(_mm256_add_ps(ft_b, fs_b), half));

        gm_r = _mm256_add_ps(_mm256_mul_ps(gm_r, fta), _mm256_mul_ps(ft_r, fit_a));
        gm_g = _mm256_add_ps(_mm256_mul_ps(gm_g, fta), _mm256_mul_ps(ft_g, fit_a));
        gm_b = _mm256_add_ps(_mm256_mul_ps(gm_b, fta), _mm256_mul_ps(ft_b, fit_a));

        gm_r = _mm256_add_ps(_mm256_mul_ps(gm_r, invw), _mm256_mul_ps(fs_r, w));
        gm_g = _mm256_add_ps(_mm256_mul_ps(gm_g, invw), _mm256_mul_ps(fs_g, w));
        gm_b = _mm256_add_ps(_mm256_mul_ps(gm_b, invw), _mm256_mul_ps(fs_b, w));

        gm_r = nan_to_num_ps(gm_r);
        gm_g = nan_to_num_ps(gm_g);
        gm_b = nan_to_num_ps(gm_b);

        __m256 fr = _mm256_add_ps(_mm256_mul_ps(gm_r, fta), _mm256_mul_ps(fb_r, fit_a));
        __m256 fg = _mm256_add_ps(_mm256_mul_ps(gm_g, fta), _mm256_mul_ps(fb_g, fit_a));
        __m256 fb = _mm256_add_ps(_mm256_mul_ps(gm_b, fta), _mm256_mul_ps(fb_b, fit_a));

        store_unit_f32_to_u8_rgb8_avx2(fr, fg, fb, out, i);
    }

    if (i < pixels) {
        kernel_scalar_rgba(base + 3*i, texture + 4*i, skin + 4*i, im_alpha + i,
                           out + 3*i, pixels - i);
    }
}

/* ---------- SSE4.2 skeleton (process 4 pixels via manual loads) ---------- */

/* 4-lane u8->f32 [0,1] from scalar bytes (works with interleaved strides) */
static inline __m128 u8x4_to_unit_f32(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    __m128i vi = _mm_setr_epi32((int)a, (int)b, (int)c, (int)d);
    return _mm_mul_ps(_mm_cvtepi32_ps(vi), _mm_set1_ps(1.0f/255.0f));
}

static inline __m128 load4_u8_to_unit_f32(const uint8_t *p) {
    /* p[0..3] are consecutive bytes (for im_alpha) */
    __m128i v8  = _mm_cvtsi32_si128(*(const int*)p);  /* 4 bytes into xmm */
    __m128i v16 = _mm_cvtepu8_epi16(v8);              /* widen to 8 x u16, we use low 4 */
    __m128i v32 = _mm_cvtepu16_epi32(v16);
    return _mm_mul_ps(_mm_cvtepi32_ps(v32), _mm_set1_ps(1.0f/255.0f));
}

static inline __m128 clamp01_ps128(__m128 x) {
    return _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(0.0f)), _mm_set1_ps(1.0f));
}

static inline __m128 nan_to_num_ps128(__m128 x) {
    __m128 cmp = _mm_cmpord_ps(x, x); /* 0 for NaN lanes */
    return _mm_blendv_ps(_mm_set1_ps(0.0f), x, cmp);
}


static void kernel_sse42_rgb(const uint8_t *base, const uint8_t *texture,
                             const uint8_t *skin, const uint8_t *im_alpha,
                             uint8_t *out, npy_intp pixels) {
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 one  = _mm_set1_ps(1.0f);
    const __m128 w    = _mm_set1_ps((float)SKIN_WEIGHT);
    const __m128 invw = _mm_set1_ps(1.0f - (float)SKIN_WEIGHT);

    npy_intp i = 0;
    for (; i + 4 <= pixels; i += 4) {
        __m128 fb_r = u8x4_to_unit_f32(base[3*(i+0)+0], base[3*(i+1)+0],
                                       base[3*(i+2)+0], base[3*(i+3)+0]);
        __m128 fb_g = u8x4_to_unit_f32(base[3*(i+0)+1], base[3*(i+1)+1],
                                       base[3*(i+2)+1], base[3*(i+3)+1]);
        __m128 fb_b = u8x4_to_unit_f32(base[3*(i+0)+2], base[3*(i+1)+2],
                                       base[3*(i+2)+2], base[3*(i+3)+2]);

        __m128 ft_r = u8x4_to_unit_f32(texture[3*(i+0)+0], texture[3*(i+1)+0],
                                       texture[3*(i+2)+0], texture[3*(i+3)+0]);
        __m128 ft_g = u8x4_to_unit_f32(texture[3*(i+0)+1], texture[3*(i+1)+1],
                                       texture[3*(i+2)+1], texture[3*(i+3)+1]);
        __m128 ft_b = u8x4_to_unit_f32(texture[3*(i+0)+2], texture[3*(i+1)+2],
                                       texture[3*(i+2)+2], texture[3*(i+3)+2]);

        __m128 fs_r = u8x4_to_unit_f32(skin[4*(i+0)+0], skin[4*(i+1)+0],
                                       skin[4*(i+2)+0], skin[4*(i+3)+0]);
        __m128 fs_g = u8x4_to_unit_f32(skin[4*(i+0)+1], skin[4*(i+1)+1],
                                       skin[4*(i+2)+1], skin[4*(i+3)+1]);
        __m128 fs_b = u8x4_to_unit_f32(skin[4*(i+0)+2], skin[4*(i+1)+2],
                                       skin[4*(i+2)+2], skin[4*(i+3)+2]);

        __m128 fa_im = load4_u8_to_unit_f32(im_alpha + i);
        __m128 fit_a = _mm_sub_ps(one, fa_im);

        __m128 gm_r = clamp01_ps128(_mm_sub_ps(_mm_add_ps(ft_r, fs_r), half));
        __m128 gm_g = clamp01_ps128(_mm_sub_ps(_mm_add_ps(ft_g, fs_g), half));
        __m128 gm_b = clamp01_ps128(_mm_sub_ps(_mm_add_ps(ft_b, fs_b), half));

        gm_r = _mm_add_ps(_mm_mul_ps(gm_r, fa_im), _mm_mul_ps(ft_r, fit_a));
        gm_g = _mm_add_ps(_mm_mul_ps(gm_g, fa_im), _mm_mul_ps(ft_g, fit_a));
        gm_b = _mm_add_ps(_mm_mul_ps(gm_b, fa_im), _mm_mul_ps(ft_b, fit_a));

        gm_r = _mm_add_ps(_mm_mul_ps(gm_r, invw), _mm_mul_ps(fs_r, w));
        gm_g = _mm_add_ps(_mm_mul_ps(gm_g, invw), _mm_mul_ps(fs_g, w));
        gm_b = _mm_add_ps(_mm_mul_ps(gm_b, invw), _mm_mul_ps(fs_b, w));

        gm_r = nan_to_num_ps128(gm_r);
        gm_g = nan_to_num_ps128(gm_g);
        gm_b = nan_to_num_ps128(gm_b);

        __m128 fr = _mm_add_ps(_mm_mul_ps(gm_r, fa_im), _mm_mul_ps(fb_r, fit_a));
        __m128 fg = _mm_add_ps(_mm_mul_ps(gm_g, fa_im), _mm_mul_ps(fb_g, fit_a));
        __m128 fb = _mm_add_ps(_mm_mul_ps(gm_b, fa_im), _mm_mul_ps(fb_b, fit_a));

        float rr[4], gg[4], bb[4];
        _mm_storeu_ps(rr, fr);
        _mm_storeu_ps(gg, fg);
        _mm_storeu_ps(bb, fb);

        for (int k = 0; k < 4; ++k) {
            int r = (int)(rr[k] * 255.0f);
            int g = (int)(gg[k] * 255.0f);
            int b = (int)(bb[k] * 255.0f);
            out[3*(i+k)+0] = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
            out[3*(i+k)+1] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
            out[3*(i+k)+2] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
        }
    }

    if (i < pixels) {
        kernel_scalar_rgb(base + 3*i, texture + 3*i, skin + 4*i, im_alpha + i,
                          out + 3*i, pixels - i);
    }
}

static void kernel_sse42_rgba(const uint8_t *base, const uint8_t *texture,
                              const uint8_t *skin, const uint8_t *im_alpha,
                              uint8_t *out, npy_intp pixels) {
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 one  = _mm_set1_ps(1.0f);
    const __m128 w    = _mm_set1_ps((float)SKIN_WEIGHT);
    const __m128 invw = _mm_set1_ps(1.0f - (float)SKIN_WEIGHT);

    npy_intp i = 0;
    for (; i + 4 <= pixels; i += 4) {
        __m128 fb_r = u8x4_to_unit_f32(base[3*(i+0)+0], base[3*(i+1)+0],
                                       base[3*(i+2)+0], base[3*(i+3)+0]);
        __m128 fb_g = u8x4_to_unit_f32(base[3*(i+0)+1], base[3*(i+1)+1],
                                       base[3*(i+2)+1], base[3*(i+3)+1]);
        __m128 fb_b = u8x4_to_unit_f32(base[3*(i+0)+2], base[3*(i+1)+2],
                                       base[3*(i+2)+2], base[3*(i+3)+2]);

        __m128 ft_r = u8x4_to_unit_f32(texture[4*(i+0)+0], texture[4*(i+1)+0],
                                       texture[4*(i+2)+0], texture[4*(i+3)+0]);
        __m128 ft_g = u8x4_to_unit_f32(texture[4*(i+0)+1], texture[4*(i+1)+1],
                                       texture[4*(i+2)+1], texture[4*(i+3)+1]);
        __m128 ft_b = u8x4_to_unit_f32(texture[4*(i+0)+2], texture[4*(i+1)+2],
                                       texture[4*(i+2)+2], texture[4*(i+3)+2]);
        __m128 ft_a = u8x4_to_unit_f32(texture[4*(i+0)+3], texture[4*(i+1)+3],
                                       texture[4*(i+2)+3], texture[4*(i+3)+3]);

        __m128 fs_r = u8x4_to_unit_f32(skin[4*(i+0)+0], skin[4*(i+1)+0],
                                       skin[4*(i+2)+0], skin[4*(i+3)+0]);
        __m128 fs_g = u8x4_to_unit_f32(skin[4*(i+0)+1], skin[4*(i+1)+1],
                                       skin[4*(i+2)+1], skin[4*(i+3)+1]);
        __m128 fs_b = u8x4_to_unit_f32(skin[4*(i+0)+2], skin[4*(i+1)+2],
                                       skin[4*(i+2)+2], skin[4*(i+3)+2]);

        __m128 fa_im = load4_u8_to_unit_f32(im_alpha + i);
        __m128 fta   = _mm_mul_ps(ft_a, fa_im);   /* texture_alpha */
        __m128 fit_a = _mm_sub_ps(one, fta);

        __m128 gm_r = clamp01_ps128(_mm_sub_ps(_mm_add_ps(ft_r, fs_r), half));
        __m128 gm_g = clamp01_ps128(_mm_sub_ps(_mm_add_ps(ft_g, fs_g), half));
        __m128 gm_b = clamp01_ps128(_mm_sub_ps(_mm_add_ps(ft_b, fs_b), half));

        gm_r = _mm_add_ps(_mm_mul_ps(gm_r, fta), _mm_mul_ps(ft_r, fit_a));
        gm_g = _mm_add_ps(_mm_mul_ps(gm_g, fta), _mm_mul_ps(ft_g, fit_a));
        gm_b = _mm_add_ps(_mm_mul_ps(gm_b, fta), _mm_mul_ps(ft_b, fit_a));

        gm_r = _mm_add_ps(_mm_mul_ps(gm_r, invw), _mm_mul_ps(fs_r, w));
        gm_g = _mm_add_ps(_mm_mul_ps(gm_g, invw), _mm_mul_ps(fs_g, w));
        gm_b = _mm_add_ps(_mm_mul_ps(gm_b, invw), _mm_mul_ps(fs_b, w));

        gm_r = nan_to_num_ps128(gm_r);
        gm_g = nan_to_num_ps128(gm_g);
        gm_b = nan_to_num_ps128(gm_b);

        __m128 fr = _mm_add_ps(_mm_mul_ps(gm_r, fta), _mm_mul_ps(fb_r, fit_a));
        __m128 fg = _mm_add_ps(_mm_mul_ps(gm_g, fta), _mm_mul_ps(fb_g, fit_a));
        __m128 fb = _mm_add_ps(_mm_mul_ps(gm_b, fta), _mm_mul_ps(fb_b, fit_a));

        float rr[4], gg[4], bb[4];
        _mm_storeu_ps(rr, fr);
        _mm_storeu_ps(gg, fg);
        _mm_storeu_ps(bb, fb);

        for (int k = 0; k < 4; ++k) {
            int r = (int)(rr[k] * 255.0f);
            int g = (int)(gg[k] * 255.0f);
            int b = (int)(bb[k] * 255.0f);
            out[3*(i+k)+0] = (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
            out[3*(i+k)+1] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : g);
            out[3*(i+k)+2] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
        }
    }

    if (i < pixels) {
        kernel_scalar_rgba(base + 3*i, texture + 4*i, skin + 4*i, im_alpha + i,
                           out + 3*i, pixels - i);
    }
}


/* ---------- Kernel dispatch ---------- */

static kernel_kind pick_kernel(const char *force_name) {
    if (force_name) {
        if (strcmp(force_name, "scalar") == 0) return KERNEL_SCALAR;
        if (strcmp(force_name, "sse42")  == 0) return KERNEL_SSE42;
        if (strcmp(force_name, "avx2")   == 0) return KERNEL_AVX2;
        if (strcmp(force_name, "auto")   == 0) {/* fall through */}
    }
    /* Auto: prefer AVX2, then SSE4.2, else scalar */
    if (cpu_supports_avx2() && os_supports_avx()) return KERNEL_AVX2;
    if (cpu_supports_sse42()) return KERNEL_SSE42;
    return KERNEL_SCALAR;
}

/* ---------- Python binding ---------- */

/* Convert base (H,W,3 or H,W,4) -> packed RGB (H,W,3). Returns a NEW ref.
   If base is already (H,W,3), this returns a new C-contig copy of it (to be safe). */
static PyArrayObject* ensure_base_rgb(PyArrayObject *base_in, const char *name) {
    if (PyArray_NDIM(base_in) != 3) {
        PyErr_Format(PyExc_ValueError, "%s must have shape (H, W, 3) or (H, W, 4)", name);
        return NULL;
    }
    npy_intp const *dims_in = PyArray_DIMS(base_in);
    npy_intp H = dims_in[0], W = dims_in[1], C = dims_in[2];
    if (!(C == 3 || C == 4)) {
        PyErr_Format(PyExc_ValueError, "%s must have 3 or 4 channels", name);
        return NULL;
    }

    /* Always produce a fresh C-contiguous uint8 (H,W,3) we own. */
    npy_intp dims_out[3] = {H, W, 3};
    PyArrayObject *base_rgb = (PyArrayObject*)PyArray_SimpleNew(3, dims_out, NPY_UINT8);
    if (!base_rgb) return NULL;

    const uint8_t *src = (const uint8_t*)PyArray_DATA(base_in);
    uint8_t *dst       = (uint8_t*)PyArray_DATA(base_rgb);
    const npy_intp pixels = H * W;

    if (C == 3) {
        /* Packed copy */
        memcpy(dst, src, (size_t)(pixels * 3));
        return base_rgb;
    }

    /* C == 4: strip alpha, keep RGB packed */
    for (npy_intp i = 0; i < pixels; ++i) {
        dst[3*i + 0] = src[4*i + 0];
        dst[3*i + 1] = src[4*i + 1];
        dst[3*i + 2] = src[4*i + 2];
    }
    return base_rgb;
}

static PyObject* py_normal_grain_merge(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"base", "texture", "skin", "im_alpha", "kernel", NULL};

    PyObject *base_obj = NULL, *texture_obj = NULL, *skin_obj = NULL, *im_alpha_obj = NULL;
    const char *kernel_name = "auto";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|s", kwlist,
                                     &base_obj, &texture_obj, &skin_obj, &im_alpha_obj,
                                     &kernel_name)) {
        return NULL;
    }

    /* Materialize arrays we own. Do NOT decref the *_obj borrowed refs. */
    /* Borrowed -> owned, uint8, C-contig (you already have get_uint8_c_contig) */
    PyArrayObject *base_u8 = NULL, *texture = NULL, *skin = NULL, *im_alpha = NULL;
    if (!get_uint8_c_contig(base_obj, &base_u8, "base") ||
        !get_uint8_c_contig(texture_obj, &texture, "texture") ||
        !get_uint8_c_contig(skin_obj, &skin, "skin") ||
        !get_uint8_c_contig(im_alpha_obj, &im_alpha, "im_alpha")) {
        Py_XDECREF(base_u8); Py_XDECREF(texture); Py_XDECREF(skin); Py_XDECREF(im_alpha);
        return NULL;
    }

    /* If base is RGBA, pack to RGB; if it’s already RGB, make a packed copy */
    PyArrayObject *base = ensure_base_rgb(base_u8, "base");
    if (!base) {
        Py_DECREF(base_u8); Py_DECREF(texture); Py_DECREF(skin); Py_DECREF(im_alpha);
        return NULL;
    }
    Py_DECREF(base_u8);  /* drop the intermediate reference, we own `base` now */

    int texture_has_alpha = 0;
    npy_intp H = 0, W = 0;
    if (!check_shape_requirements(base, texture, skin, im_alpha,
                                  &texture_has_alpha, &H, &W)) {
        Py_DECREF(base); Py_DECREF(texture); Py_DECREF(skin); Py_DECREF(im_alpha);
        return NULL;
    }

    /* Allocate output (H, W, 3) uint8 */
    PyObject *out = PyArray_NewLikeArray(base, NPY_ANYORDER, NULL, 0);
    if (!out) {
        Py_XDECREF(base); Py_XDECREF(texture); Py_XDECREF(skin); Py_XDECREF(im_alpha);
        return NULL;
    }

    const uint8_t *p_base    = (const uint8_t*)PyArray_DATA(base);
    const uint8_t *p_texture = (const uint8_t*)PyArray_DATA(texture);
    const uint8_t *p_skin    = (const uint8_t*)PyArray_DATA(skin);
    const uint8_t *p_imalpha = (const uint8_t*)PyArray_DATA(im_alpha);
    uint8_t *p_out           = (uint8_t*)PyArray_DATA((PyArrayObject*)out);

    const npy_intp pixels = H * W;

    kernel_kind k = pick_kernel(kernel_name);

    /* Optional: release the GIL around pure C loops. No Python API calls inside kernels. */
    NPY_BEGIN_ALLOW_THREADS

    if (!texture_has_alpha) {
        if (k == KERNEL_AVX2) {
            kernel_avx2_rgb(p_base, p_texture, p_skin, p_imalpha, p_out, pixels);
        } else if (k == KERNEL_SSE42) {
            kernel_sse42_rgb(p_base, p_texture, p_skin, p_imalpha, p_out, pixels);
        } else {
            kernel_scalar_rgb(p_base, p_texture, p_skin, p_imalpha, p_out, pixels);
        }
    } else {
        if (k == KERNEL_AVX2) {
            kernel_avx2_rgba(p_base, p_texture, p_skin, p_imalpha, p_out, pixels);
        } else if (k == KERNEL_SSE42) {
            kernel_sse42_rgba(p_base, p_texture, p_skin, p_imalpha, p_out, pixels);
        } else {
            kernel_scalar_rgba(p_base, p_texture, p_skin, p_imalpha, p_out, pixels);
        }
    }

    NPY_END_ALLOW_THREADS

    /* DECREF only what we own. */
    Py_DECREF(base); Py_DECREF(texture); Py_DECREF(skin); Py_DECREF(im_alpha);
    return out;
}

static PyMethodDef Methods[] = {
    {"normal_grain_merge", (PyCFunction)py_normal_grain_merge, METH_VARARGS | METH_KEYWORDS,
     "normal_grain_merge(base, texture, skin, im_alpha, kernel='auto') -> np.ndarray\n"
     "kernel: 'auto', 'scalar', 'sse42', or 'avx2'"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "normal_grain_merge",
    "Normal Grain Merge Module",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_normal_grain_merge(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
