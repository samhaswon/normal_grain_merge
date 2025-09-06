#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <immintrin.h>  /* AVX2 + SSE4.2 */

#if defined(__GNUC__) || defined(__clang__)
#define HAVE_BUILTIN_CPU_SUPPORTS 1
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
static int get_uint8_c_contig(PyObject *obj, PyArrayObject **out, const char *name) {
    const int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;
    PyArrayObject *arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_UINT8, flags);
    if (!arr) {
        PyErr_Format(PyExc_TypeError, "%s must be a uint8 ndarray", name);
        return 0;
    }
    *out = arr;  /* new reference */
    return 1;
}

static int ensure_uint8_contig(PyArrayObject **arr, const char *name) {
    PyArrayObject *tmp = (PyArrayObject*)PyArray_FROM_OTF(
        (PyObject*)(*arr), NPY_UINT8, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!tmp) return 0;
    Py_XDECREF(*arr);
    *arr = tmp;
    return 1;
}

static int check_shape_requirements(PyArrayObject *base,
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

static inline __m256 gather_u8_to_f32_avx2(const uint8_t *base_ptr, int stride, npy_intp start_idx) {
    /* indices: start_idx .. start_idx+7, each multiplied by stride */
    const int idx0 = (int)((start_idx + 0) * stride);
    const int idx1 = (int)((start_idx + 1) * stride);
    const int idx2 = (int)((start_idx + 2) * stride);
    const int idx3 = (int)((start_idx + 3) * stride);
    const int idx4 = (int)((start_idx + 4) * stride);
    const int idx5 = (int)((start_idx + 5) * stride);
    const int idx6 = (int)((start_idx + 6) * stride);
    const int idx7 = (int)((start_idx + 7) * stride);

    __m256i offs = _mm256_setr_epi32(idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7);
    __m256i v32  = _mm256_i32gather_epi32((const int*)base_ptr, offs, 1); /* gather 8 x u8, read as u32 */
    v32 = _mm256_and_si256(v32, _mm256_set1_epi32(0xFF));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(v32), _mm256_set1_ps(1.0f/255.0f));
}

static inline __m256 clamp01_ps(__m256 x) {
    return _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(0.0f)), _mm256_set1_ps(1.0f));
}

static inline void store_f32_to_u8_3x_avx2(__m256 fr, __m256 fg, __m256 fb,
                                           uint8_t *out_ptr, npy_intp start_idx) {
    /* Convert 8-lane float [0,1] back to uint8 and scatter with stride 3 */
    __m256 scale = _mm256_set1_ps(255.0f);
    __m256i ir = _mm256_cvtps_epi32(_mm256_mul_ps(fr, scale));
    __m256i ig = _mm256_cvtps_epi32(_mm256_mul_ps(fg, scale));
    __m256i ib = _mm256_cvtps_epi32(_mm256_mul_ps(fb, scale));

    int r[8], g[8], b[8];
    _mm256_storeu_si256((__m256i*)r, ir);
    _mm256_storeu_si256((__m256i*)g, ig);
    _mm256_storeu_si256((__m256i*)b, ib);

    for (int k = 0; k < 8; ++k) {
        npy_intp p = start_idx + k;
        out_ptr[3*p+0] = (uint8_t)((r[k] < 0 ? 0 : r[k] > 255 ? 255 : r[k]));
        out_ptr[3*p+1] = (uint8_t)((g[k] < 0 ? 0 : g[k] > 255 ? 255 : g[k]));
        out_ptr[3*p+2] = (uint8_t)((b[k] < 0 ? 0 : b[k] > 255 ? 255 : b[k]));
    }
}

static void kernel_avx2_rgb(const uint8_t *base, const uint8_t *texture,
                            const uint8_t *skin, const uint8_t *im_alpha,
                            uint8_t *out, npy_intp pixels) {
    const int stride3 = 3, stride4 = 4;
    npy_intp i = 0;
    for (; i + 8 <= pixels; i += 8) {
        __m256 fb_r = gather_u8_to_f32_avx2(base+0, stride3, i);
        __m256 fb_g = gather_u8_to_f32_avx2(base+1, stride3, i);
        __m256 fb_b = gather_u8_to_f32_avx2(base+2, stride3, i);

        __m256 ft_r = gather_u8_to_f32_avx2(texture+0, stride3, i);
        __m256 ft_g = gather_u8_to_f32_avx2(texture+1, stride3, i);
        __m256 ft_b = gather_u8_to_f32_avx2(texture+2, stride3, i);

        __m256 fs_r = gather_u8_to_f32_avx2(skin+0, stride4, i);
        __m256 fs_g = gather_u8_to_f32_avx2(skin+1, stride4, i);
        __m256 fs_b = gather_u8_to_f32_avx2(skin+2, stride4, i);
        __m256 fs_a = gather_u8_to_f32_avx2(skin+3, stride4, i);

        /* im_alpha scalar path for now */
        float a_im_s[8];
        for (int k = 0; k < 8; ++k) a_im_s[k] = im_alpha[i+k] * (1.0f/255.0f);
        __m256 fa_im = _mm256_loadu_ps(a_im_s);

        /* Placeholder math: out = base */
        __m256 fr = fb_r, fg = fb_g, fbv = fb_b;

        fr = clamp01_ps(fr); fg = clamp01_ps(fg); fbv = clamp01_ps(fbv);
        store_f32_to_u8_3x_avx2(fr, fg, fbv, out, i);
    }

    /* Tail */
    if (i < pixels) {
        kernel_scalar_rgb(base + 3*i, texture + 3*i, skin + 4*i, im_alpha + i,
                          out + 3*i, pixels - i);
    }
}

static void kernel_avx2_rgba(const uint8_t *base, const uint8_t *texture,
                             const uint8_t *skin, const uint8_t *im_alpha,
                             uint8_t *out, npy_intp pixels) {
    const int stride3 = 3, stride4 = 4;
    npy_intp i = 0;
    for (; i + 8 <= pixels; i += 8) {
        __m256 fb_r = gather_u8_to_f32_avx2(base+0, stride3, i);
        __m256 fb_g = gather_u8_to_f32_avx2(base+1, stride3, i);
        __m256 fb_b = gather_u8_to_f32_avx2(base+2, stride3, i);

        __m256 ft_r = gather_u8_to_f32_avx2(texture+0, stride4, i);
        __m256 ft_g = gather_u8_to_f32_avx2(texture+1, stride4, i);
        __m256 ft_b = gather_u8_to_f32_avx2(texture+2, stride4, i);
        __m256 ft_a = gather_u8_to_f32_avx2(texture+3, stride4, i);

        __m256 fs_r = gather_u8_to_f32_avx2(skin+0, stride4, i);
        __m256 fs_g = gather_u8_to_f32_avx2(skin+1, stride4, i);
        __m256 fs_b = gather_u8_to_f32_avx2(skin+2, stride4, i);
        __m256 fs_a = gather_u8_to_f32_avx2(skin+3, stride4, i);

        float a_im_s[8];
        for (int k = 0; k < 8; ++k) a_im_s[k] = im_alpha[i+k] * (1.0f/255.0f);
        __m256 fa_im = _mm256_loadu_ps(a_im_s);

        /* Placeholder math: out = base */
        __m256 fr = fb_r, fg = fb_g, fbv = fb_b;

        fr = clamp01_ps(fr); fg = clamp01_ps(fg); fbv = clamp01_ps(fbv);
        store_f32_to_u8_3x_avx2(fr, fg, fbv, out, i);
    }
    if (i < pixels) {
        kernel_scalar_rgba(base + 3*i, texture + 4*i, skin + 4*i, im_alpha + i,
                           out + 3*i, pixels - i);
    }
}

/* ---------- SSE4.2 skeleton (process 4 pixels via manual loads) ---------- */

static inline __m128 u8x4_to_f32_sse(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    __m128i vi = _mm_setr_epi32((int)a, (int)b, (int)c, (int)d);
    return _mm_mul_ps(_mm_cvtepi32_ps(vi), _mm_set1_ps(1.0f/255.0f));
}

static void kernel_sse42_rgb(const uint8_t *base, const uint8_t *texture,
                             const uint8_t *skin, const uint8_t *im_alpha,
                             uint8_t *out, npy_intp pixels) {
    npy_intp i = 0;
    for (; i + 4 <= pixels; i += 4) {
        /* Gather each channel manually for 4 pixels */
        __m128 fb_r = u8x4_to_f32_sse(base[3*(i+0)+0], base[3*(i+1)+0],
                                      base[3*(i+2)+0], base[3*(i+3)+0]);
        __m128 fb_g = u8x4_to_f32_sse(base[3*(i+0)+1], base[3*(i+1)+1],
                                      base[3*(i+2)+1], base[3*(i+3)+1]);
        __m128 fb_b = u8x4_to_f32_sse(base[3*(i+0)+2], base[3*(i+1)+2],
                                      base[3*(i+2)+2], base[3*(i+3)+2]);

        __m128 ft_r = u8x4_to_f32_sse(texture[3*(i+0)+0], texture[3*(i+1)+0],
                                      texture[3*(i+2)+0], texture[3*(i+3)+0]);
        __m128 ft_g = u8x4_to_f32_sse(texture[3*(i+0)+1], texture[3*(i+1)+1],
                                      texture[3*(i+2)+1], texture[3*(i+3)+1]);
        __m128 ft_b = u8x4_to_f32_sse(texture[3*(i+0)+2], texture[3*(i+1)+2],
                                      texture[3*(i+2)+2], texture[3*(i+3)+2]);

        __m128 fs_r = u8x4_to_f32_sse(skin[4*(i+0)+0], skin[4*(i+1)+0],
                                      skin[4*(i+2)+0], skin[4*(i+3)+0]);
        __m128 fs_g = u8x4_to_f32_sse(skin[4*(i+0)+1], skin[4*(i+1)+1],
                                      skin[4*(i+2)+1], skin[4*(i+3)+1]);
        __m128 fs_b = u8x4_to_f32_sse(skin[4*(i+0)+2], skin[4*(i+1)+2],
                                      skin[4*(i+2)+2], skin[4*(i+3)+2]);
        __m128 fs_a = u8x4_to_f32_sse(skin[4*(i+0)+3], skin[4*(i+1)+3],
                                      skin[4*(i+2)+3], skin[4*(i+3)+3]);

        float a_im_s[4];
        for (int k = 0; k < 4; ++k) a_im_s[k] = im_alpha[i+k] * (1.0f/255.0f);
        __m128 fa_im = _mm_loadu_ps(a_im_s);

        __m128 fr = fb_r, fg = fb_g, fbv = fb_b;  /* placeholder */

        /* Store back */
        float rr[4], gg[4], bb[4];
        _mm_storeu_ps(rr, fr);
        _mm_storeu_ps(gg, fg);
        _mm_storeu_ps(bb, fbv);
        for (int k = 0; k < 4; ++k) {
            int r = (int)(rr[k]*255.0f + 0.5f);
            int g = (int)(gg[k]*255.0f + 0.5f);
            int b = (int)(bb[k]*255.0f + 0.5f);
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
    npy_intp i = 0;
    for (; i + 4 <= pixels; i += 4) {
        __m128 fb_r = u8x4_to_f32_sse(base[3*(i+0)+0], base[3*(i+1)+0],
                                      base[3*(i+2)+0], base[3*(i+3)+0]);
        __m128 fb_g = u8x4_to_f32_sse(base[3*(i+0)+1], base[3*(i+1)+1],
                                      base[3*(i+2)+1], base[3*(i+3)+1]);
        __m128 fb_b = u8x4_to_f32_sse(base[3*(i+0)+2], base[3*(i+1)+2],
                                      base[3*(i+2)+2], base[3*(i+3)+2]);

        __m128 ft_r = u8x4_to_f32_sse(texture[4*(i+0)+0], texture[4*(i+1)+0],
                                      texture[4*(i+2)+0], texture[4*(i+3)+0]);
        __m128 ft_g = u8x4_to_f32_sse(texture[4*(i+0)+1], texture[4*(i+1)+1],
                                      texture[4*(i+2)+1], texture[4*(i+3)+1]);
        __m128 ft_b = u8x4_to_f32_sse(texture[4*(i+0)+2], texture[4*(i+1)+2],
                                      texture[4*(i+2)+2], texture[4*(i+3)+2]);
        __m128 ft_a = u8x4_to_f32_sse(texture[4*(i+0)+3], texture[4*(i+1)+3],
                                      texture[4*(i+2)+3], texture[4*(i+3)+3]);

        __m128 fs_r = u8x4_to_f32_sse(skin[4*(i+0)+0], skin[4*(i+1)+0],
                                      skin[4*(i+2)+0], skin[4*(i+3)+0]);
        __m128 fs_g = u8x4_to_f32_sse(skin[4*(i+0)+1], skin[4*(i+1)+1],
                                      skin[4*(i+2)+1], skin[4*(i+3)+1]);
        __m128 fs_b = u8x4_to_f32_sse(skin[4*(i+0)+2], skin[4*(i+1)+2],
                                      skin[4*(i+2)+2], skin[4*(i+3)+2]);
        __m128 fs_a = u8x4_to_f32_sse(skin[4*(i+0)+3], skin[4*(i+1)+3],
                                      skin[4*(i+2)+3], skin[4*(i+3)+3]);

        float a_im_s[4];
        for (int k = 0; k < 4; ++k) a_im_s[k] = im_alpha[i+k] * (1.0f/255.0f);
        __m128 fa_im = _mm_loadu_ps(a_im_s);

        __m128 fr = fb_r, fg = fb_g, fbv = fb_b;  /* placeholder */

        float rr[4], gg[4], bb[4];
        _mm_storeu_ps(rr, fr);
        _mm_storeu_ps(gg, fg);
        _mm_storeu_ps(bb, fbv);
        for (int k = 0; k < 4; ++k) {
            int r = (int)(rr[k]*255.0f + 0.5f);
            int g = (int)(gg[k]*255.0f + 0.5f);
            int b = (int)(bb[k]*255.0f + 0.5f);
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
        if (strcmp(force_name, "auto")   == 0) return KERNEL_AUTO;
    }
#if HAVE_BUILTIN_CPU_SUPPORTS
    if (!force_name || strcmp(force_name, "auto") == 0) {
        if (__builtin_cpu_supports("avx2")) return KERNEL_AVX2;
        if (__builtin_cpu_supports("sse4.2")) return KERNEL_SSE42;
        return KERNEL_SCALAR;
    }
#endif
    return KERNEL_SCALAR;
}

/* ---------- Python binding ---------- */

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
    PyArrayObject *base = NULL, *texture = NULL, *skin = NULL, *im_alpha = NULL;
    if (!get_uint8_c_contig(base_obj, &base, "base") ||
        !get_uint8_c_contig(texture_obj, &texture, "texture") ||
        !get_uint8_c_contig(skin_obj, &skin, "skin") ||
        !get_uint8_c_contig(im_alpha_obj, &im_alpha, "im_alpha")) {
        Py_XDECREF(base); Py_XDECREF(texture); Py_XDECREF(skin); Py_XDECREF(im_alpha);
        return NULL;
    }

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
