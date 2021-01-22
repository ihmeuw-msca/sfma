#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "complex.h"


/*
 * log_erfc.c
 * This is the C code for creating your own
 * NumPy ufunc for a log_erfc function.
 *
 * Each function of the form type_log_erfc defines the
 * log_erfc function for a different numpy dtype. Each
 * of these functions must be modified when you
 * create your own ufunc. The computations that must
 * be replaced to create a ufunc for
 * a different function are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 *
 */


static PyMethodDef LogErfcMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definitions must precede the PyMODINIT_FUNC. */

static void double_log_erfc(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double tmp;
    double sqrt_pi = 1.7724538509055159;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(double *)in;
        if (tmp > 25.0) {
            *((double *)out) = -tmp*tmp + \
                log((1.0 - 0.5/(tmp*tmp))/(tmp*sqrt_pi));
        } else {
            *((double *)out) = log(erfc(tmp));
        }
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void cdouble_log_erfc(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double complex tmp;
    double tmp_r;
    double tmp_i;
    double sqrt_pi = 1.7724538509055159;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp = *(double complex *)in;
        tmp_r = creal(tmp);
        tmp_i = cimag(tmp);
        if (tmp_r > 25.0) {
            *((double complex *)out) = -tmp*tmp + \
                log((1.0 - 0.5/(tmp*tmp))/(tmp*sqrt_pi)) -
                I*tmp_i*(2.0*tmp_r + 1.0/tmp_r - 1.0/(tmp_r*tmp_r*tmp_r));
        } else {
            *((double complex *)out) = log(erfc(tmp_r)) - \
                I*tmp_i*2.0*exp(-tmp_r*tmp_r)/(erfc(tmp_r)*sqrt_pi);
        }
        /*END main ufunc computation*/
        in += in_step;
        out += out_step;
    }
}

static void double_special(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double x;
    double sqrt_pi = 1.7724538509055159;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        x = *(double *)in;
        if (x > 25.0) {
            *((double *)out) = -log(sqrt_pi) - 0.5/(x*x);
        } else {
            *((double *)out) = x*x + log(erfc(x)*fabs(x));
        }
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void cdouble_special(char **args, npy_intp *dimensions,
                         npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double complex x;
    double x_r;
    double x_i;
    double sqrt_pi = 1.7724538509055159;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        x = *(double complex *)in;
        x_r = creal(x);
        x_i = cimag(x);
        if (x_r > 25.0) {
            *((double complex *)out) = -log(sqrt_pi) - 0.5/(x_r*x_r) + \
                I*x_i/(x_r*x_r*x_r);
        } else {
            *((double complex *)out) = x_r*x_r + log(erfc(x_r)*fabs(x_r)) + \
                I*x_i*(2.0*x_r + (-2.0*exp(-x_r*x_r)*x_r/sqrt_pi + \
                    erfc(x_r))/(x_r*erfc(x_r)));
        }
        /*END main ufunc computation*/
        in += in_step;
        out += out_step;
    }
}


/*This gives pointers to the above functions*/
PyUFuncGenericFunction log_erfc_funcs[2] = {&double_log_erfc,
                                &cdouble_log_erfc,};

static char log_erfc_types[4] = {NPY_DOUBLE,NPY_DOUBLE,
                        NPY_CDOUBLE,NPY_CDOUBLE};
static void *log_erfc_data[2] = {NULL, NULL};

PyUFuncGenericFunction special_funcs[2] = {&double_special,
                                &cdouble_special,};

static char special_types[4] = {NPY_DOUBLE,NPY_DOUBLE,
                        NPY_CDOUBLE,NPY_CDOUBLE};
static void *special_data[2] = {NULL, NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    LogErfcMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *log_erfc, *special, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    log_erfc = PyUFunc_FromFuncAndData(log_erfc_funcs,
                                    log_erfc_data,
                                    log_erfc_types, 2, 1, 1,
                                    PyUFunc_None, "log_erfc",
                                    "log_erfc_docstring", 0);
    special = PyUFunc_FromFuncAndData(special_funcs,
                                    special_data,
                                    special_types, 2, 1, 1,
                                    PyUFunc_None, "special",
                                    "special_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "log_erfc", log_erfc);
    PyDict_SetItemString(d, "special", special);
    Py_DECREF(log_erfc);
    Py_DECREF(special);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *log_erfc, *special, *d;


    m = Py_InitModule("npufunc", LogErfcMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    log_erfc = PyUFunc_FromFuncAndData(log_erfc_funcs,
                                    log_erfc_data,
                                    log_erfc_types, 2, 1, 1,
                                    PyUFunc_None, "log_erfc",
                                    "log_erfc_docstring", 0);
    special = PyUFunc_FromFuncAndData(special_funcs,
                                    special_data,
                                    special_types, 2, 1, 1,
                                    PyUFunc_None, "special",
                                    "special_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "log_erfc", log_erfc);
    PyDict_SetItemString(d, "special", special);
    Py_DECREF(log_erfc);
    Py_DECREF(special);
}
#endif
