#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

static int check_array(PyArrayObject* arr, int expected_ndim, int expected_channels, const char* name) {
    if (PyArray_NDIM(arr) != expected_ndim) {
        PyErr_Format(PyExc_ValueError, "%s must have %d dimensions", name, expected_ndim);
        return 0;
    }
    if (expected_channels > 0) {
        npy_intp* dims = PyArray_DIMS(arr);
        if (dims[2] != expected_channels) {
            PyErr_Format(PyExc_ValueError, "%s must have %d channels", name, expected_channels);
            return 0;
        }
    }
    return 1;
}

static int check_same_size(PyArrayObject* a, PyArrayObject* b, const char* name_a, const char* name_b) {
    npy_intp* dims_a = PyArray_DIMS(a);
    npy_intp* dims_b = PyArray_DIMS(b);

    if (dims_a[0] != dims_b[0] || dims_a[1] != dims_b[1]) {
        PyErr_Format(PyExc_ValueError, "%s and %s must have the same height and width", name_a, name_b);
        return 0;
    }
    return 1;
}

static PyObject* normal_grain_merge(PyObject* self, PyObject* args) {
    PyArrayObject *base, *texture, *skin, *im_alpha;

    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                          &PyArray_Type, &base,
                          &PyArray_Type, &texture,
                          &PyArray_Type, &skin,
                          &PyArray_Type, &im_alpha)) {
        return NULL;
    }

    // Type check: all must be uint8
    if (PyArray_TYPE(base) != NPY_UINT8 ||
        PyArray_TYPE(texture) != NPY_UINT8 ||
        PyArray_TYPE(skin) != NPY_UINT8 ||
        PyArray_TYPE(im_alpha) != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError, "All arrays must be of type uint8");
        return NULL;
    }

    // Shape checks
    if (!check_array(base, 3, 3, "base")) return NULL;
    if (PyArray_NDIM(texture) != 3 ||
        !(PyArray_DIMS(texture)[2] == 3 || PyArray_DIMS(texture)[2] == 4)) {
        PyErr_SetString(PyExc_ValueError, "texture must have 3 or 4 channels");
        return NULL;
    }
    if (!check_array(skin, 3, 4, "skin")) return NULL;
    if (!check_array(im_alpha, 2, -1, "im_alpha")) return NULL;

    // Size compatibility
    if (!check_same_size(base, texture, "base", "texture")) return NULL;
    if (!check_same_size(base, skin, "base", "skin")) return NULL;
    if (!check_same_size(base, im_alpha, "base", "im_alpha")) return NULL;

    // Result: copy of base for now
    PyObject* result = PyArray_NewLikeArray(base, NPY_ANYORDER, NULL, 0);
    if (!result) return NULL;

    if (PyArray_CopyInto((PyArrayObject*)result, base) < 0) {
        Py_DECREF(result);
        return NULL;
    }

    return result;
}

static PyMethodDef NormalGrainMergeMethods[] = {
    {"normal_grain_merge", normal_grain_merge, METH_VARARGS,
     "Blend images using normal grain merge."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef normalgrainmergemodule = {
    PyModuleDef_HEAD_INIT,
    "normal_grain_merge",
    "Normal Grain Merge Module",
    -1,
    NormalGrainMergeMethods
};

PyMODINIT_FUNC PyInit_normal_grain_merge(void) {
    import_array();
    return PyModule_Create(&normalgrainmergemodule);
}
