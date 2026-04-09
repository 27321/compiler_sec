
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const int64_t* in_ptr2,
                       const float* in_ptr3,
                       const int64_t* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(32)
    {
        int tid = omp_get_thread_num();
        {
            std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(768L);
            float* local_buffer_data_0 = buf_local_buffer_data_0.get();
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
            {
                {
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    static WelfordHelper<float, 4096> scalar_welford_helper0(static_cast<int64_t>(768L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(48L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(16L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                            {
                                auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                                auto tmp10 = in_ptr2[static_cast<int64_t>(x0)];
                                auto tmp21 = in_ptr4[static_cast<int64_t>(x0)];
                                auto tmp1 = 30522L;
                                auto tmp2 = c10::convert<int64_t>(tmp1);
                                auto tmp3 = int64_t(tmp0 + tmp2);
                                auto tmp4 = tmp0 < 0;
                                auto tmp5 = tmp4 ? tmp3 : tmp0;
                                auto tmp6 = tmp5;
                                auto tmp7 = c10::convert<int64_t>(tmp6);
                                TORCH_CHECK((0 <= tmp7) & (tmp7 < 30522L), "index out of bounds: 0 <= tmp7 < 30522L");
                                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*tmp5), static_cast<int64_t>(16));
                                auto tmp11 = 2L;
                                auto tmp12 = c10::convert<int64_t>(tmp11);
                                auto tmp13 = int64_t(tmp10 + tmp12);
                                auto tmp14 = tmp10 < 0;
                                auto tmp15 = tmp14 ? tmp13 : tmp10;
                                auto tmp16 = tmp15;
                                auto tmp17 = c10::convert<int64_t>(tmp16);
                                TORCH_CHECK((0 <= tmp17) & (tmp17 < 2L), "index out of bounds: 0 <= tmp17 < 2L");
                                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1 + 768L*tmp15), static_cast<int64_t>(16));
                                auto tmp20 = tmp9 + tmp19;
                                auto tmp22 = 512L;
                                auto tmp23 = c10::convert<int64_t>(tmp22);
                                auto tmp24 = int64_t(tmp21 + tmp23);
                                auto tmp25 = tmp21 < 0;
                                auto tmp26 = tmp25 ? tmp24 : tmp21;
                                auto tmp27 = tmp26;
                                auto tmp28 = c10::convert<int64_t>(tmp27);
                                TORCH_CHECK((0 <= tmp28) & (tmp28 < 512L), "index out of bounds: 0 <= tmp28 < 512L");
                                auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 768L*tmp26), static_cast<int64_t>(16));
                                auto tmp31 = tmp20 + tmp30;
                                tmp31.store(local_buffer_data_0 + static_cast<int64_t>(x1));
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp31, &welford_helper0);
                            }
                        }
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, &scalar_welford_helper0);
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                    masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(16L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(local_buffer_data_0 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp1 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp4 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(768.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-12);
                            auto tmp8 = float(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp15.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                        }
                    }
                }
            }
        }
    }
}

// Python bindings to call kernel():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>

#ifndef _MSC_VER
#if __cplusplus < 202002L
// C++20 (earlier) code
// https://en.cppreference.com/w/cpp/language/attributes/likely
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1 && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}
template <> inline float parse_arg<float>(PyObject* args, size_t n) {
    auto result = PyFloat_AsDouble(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1.0 && PyErr_Occurred()))
        throw std::runtime_error("expected float arg");
    return static_cast<float>(result);
}



static PyObject* kernel_py(PyObject* self, PyObject* args) {
    try {
        if(unlikely(!PyTuple_CheckExact(args)))
            throw std::runtime_error("tuple args required");
        if(unlikely(PyTuple_GET_SIZE(args) != 11))
            throw std::runtime_error("requires 11 args");
        kernel(parse_arg<int64_t*>(args, 0), parse_arg<float*>(args, 1), parse_arg<int64_t*>(args, 2), parse_arg<float*>(args, 3), parse_arg<int64_t*>(args, 4), parse_arg<float*>(args, 5), parse_arg<float*>(args, 6), parse_arg<float*>(args, 7), parse_arg<float*>(args, 8), parse_arg<float*>(args, 9), parse_arg<float*>(args, 10)); Py_RETURN_NONE;
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"kernel", kernel_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "kernel", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_kernel(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }
    std::istringstream iss(str_addr);
    uintptr_t addr = 0;
    iss >> addr;
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    PyObject* module = PyModule_Create(&py_module);
    if (module == NULL) {
        return NULL;
    }
    #ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
    #endif
    return module;
}
