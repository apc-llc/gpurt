#ifndef CHECK_H
#define CHECK_H

#include <chrono>
#include <cstdio>
#include <thread>
#if defined(WIN32)
#define HOST_NAME_MAX 128
#include <process.h>
#include <windows.h>
#define pthread_self() GetCurrentThreadId()
#define getpid() _getpid()
#else
#include <limits.h> // HOST_NAME_MAX
#include <pthread.h>
#include <unistd.h>
#endif // WIN32

#ifdef NDEBUG

#define CUDA_ERR_CHECK(x) do { x; } while (0)

#define CU_ERR_CHECK(x) do { x; } while (0)

#define CL_ERR_CHECK(x) do { x; } while (0)

#define CURAND_ERR_CHECK(x) do { x; } while (0)

#else // NDEBUG

#define CUDA_ERR_CHECK(x)                                            \
    do { cudaError_t err = x; if (err != cudaSuccess) {              \
        char hostname[HOST_NAME_MAX] = "";                           \
        gethostname(hostname, HOST_NAME_MAX);                        \
        fprintf(stderr, "CUDA error %d \"%s\" on %s at %s:%d\n",     \
            (int)err, cudaGetErrorString(err), hostname,             \
            __FILE__, __LINE__);                                     \
        if (!getenv("FREEZE_ON_ERROR")) {                            \
            fprintf(stderr, "You may want to set "                   \
                "FREEZE_ON_ERROR environment "                       \
                "variable to debug the case\n");                     \
            exit(-1);                                                \
        }                                                            \
        else {                                                       \
            fprintf(stderr, "thread 0x%zx of pid %d @ %s "           \
                "is entering infinite loop\n",                       \
                (size_t)pthread_self(), (int)getpid(), hostname);    \
            while (1) std::this_thread::sleep_for(                   \
                std::chrono::seconds(1)); /* 1 sec */                \
        }                                                            \
    }} while (0)

#define CU_ERR_CHECK(x)                                              \
	do { CUresult err = x; if (err != CUDA_SUCCESS) {                \
        char hostname[HOST_NAME_MAX] = "";                           \
        gethostname(hostname, HOST_NAME_MAX);                        \
        fprintf(stderr, "CUDA driver error %d on %s at %s:%d\n",     \
            (int)err, hostname, __FILE__, __LINE__);                 \
        if (!getenv("FREEZE_ON_ERROR")) {                            \
            fprintf(stderr, "You may want to set "                   \
                "FREEZE_ON_ERROR environment "                       \
                "variable to debug the case\n");                     \
            exit(-1);                                                \
        }                                                            \
        else {                                                       \
            fprintf(stderr, "thread 0x%zx of pid %d @ %s "           \
                "is entering infinite loop\n",                       \
                (size_t)pthread_self(), (int)getpid(), hostname);    \
            while (1) std::this_thread::sleep_for(                   \
                std::chrono::seconds(1)); /* 1 sec */                \
        }                                                            \
    }} while (0)

#define CL_ERR_CHECK(x)                                              \
	do { cl_int err = x; if (err != CL_SUCCESS) {                    \
        char hostname[HOST_NAME_MAX] = "";                           \
        gethostname(hostname, HOST_NAME_MAX);                        \
        fprintf(stderr, "OpenCL error %d \"%s\" on %s at %s:%d\n",   \
            (int)err, CLgpu::getErrorString(err), hostname,          \
            __FILE__, __LINE__);                                     \
        if (!getenv("FREEZE_ON_ERROR")) {                            \
            fprintf(stderr, "You may want to set "                   \
                "FREEZE_ON_ERROR environment "                       \
                "variable to debug the case\n");                     \
            exit(-1);                                                \
        }                                                            \
        else {                                                       \
            fprintf(stderr, "thread 0x%zx of pid %d @ %s "           \
                "is entering infinite loop\n",                       \
                (size_t)pthread_self(), (int)getpid(), hostname);    \
            while (1) std::this_thread::sleep_for(                   \
                std::chrono::seconds(1)); /* 1 sec */                \
        }                                                            \
    }} while (0)

#define CURAND_ERR_CHECK(x)                                          \
    do { curandStatus_t err = x; if (err != CURAND_STATUS_SUCCESS) { \
        char hostname[HOST_NAME_MAX] = "";                           \
        gethostname(hostname, HOST_NAME_MAX);                        \
        fprintf(stderr, "CURAND error %d on %s at %s:%d\n",          \
            (int)err, hostname, __FILE__, __LINE__);                 \
        if (!getenv("FREEZE_ON_ERROR")) {                            \
            fprintf(stderr, "You may want to set "                   \
                "FREEZE_ON_ERROR environment "                       \
                "variable to debug the case\n");                     \
            exit(-1);                                                \
        }                                                            \
        else {                                                       \
            fprintf(stderr, "thread 0x%zx of pid %d @ %s "           \
                "is entering infinite loop\n",                       \
                (size_t)pthread_self(), (int)getpid(), hostname);    \
            while (1) std::this_thread::sleep_for(                   \
                std::chrono::seconds(1)); /* 1 sec */                \
        }                                                            \
    }} while (0)

#endif // NDEBUG

#endif // CHECK_H

