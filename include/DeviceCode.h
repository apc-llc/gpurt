#ifndef DEVICE_CODE_H
#define DEVICE_CODE_H

#if !defined(__CUDACC__)

#if !defined(__device__)
#define __device__
#endif
#if !defined(__host__)
#define __host__
#endif
#if !defined(__constant__)
#define __constant__
#endif

#define cudaError_t int
#define cudaSuccess 0

typedef struct dim3
{
    unsigned int x, y, z;

#if !defined(__OPENCL_VERSION__)
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) { }
#endif
}
dim3;

#else

#include <stdint.h>

__device__ extern uint32_t __uAtomicAdd(uint32_t*, uint32_t);
__device__ extern double __dAtomicAdd(double*, double);
__device__ extern uint32_t __uAtomicAnd(uint32_t*, uint32_t);
__device__ extern uint32_t __uAtomicCAS(uint32_t*, uint32_t, uint32_t);
__device__ extern uint32_t __uAtomicExch(uint32_t*, uint32_t);

#define threadIdx(x) threadIdx.x
#define blockIdx(x) blockIdx.x
#define blockDim(x) blockDim.x
#define gridDim(x) gridDim.x

#endif // __CUDACC__

#if defined(__OPENCL_VERSION__)

#if !defined(__global__)
#define __global__ __kernel
#endif
#if !defined(__shared__)
#define __shared__
#endif

// Do not use access attribute in OpenCL
#define PRIVATE
#define PUBLIC

typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;

#if 0
// No atomic_add(double*, double) in OpenCL - always emulate
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
double __dAtomicAdd(__global double* address, double delta)
{
	union
	{
		double f;
		unsigned long i;
	}
	old;
	union
	{
		double f;
		unsigned long i;
	}
	new1;

	do
	{
		old.f = *address;
		new1.f = old.f + delta;
	}
	while (atom_cmpxchg((volatile __global unsigned long *)address, old.i, new1.i) != old.i);

	return old.f;
}
#endif

#define __uAtomicAdd(p, val) atomic_add(p, val)
#define __uAtomicAnd(p, val) atomic_and(p, val)
#define __uAtomicCAS(p, cmp, val) atomic_cmpxchg(p, cmp, val)
#define __uAtomicExch(p, val) atomic_xchg(p, val)

#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE)
#define __threadfence() mem_fence(CLK_GLOBAL_MEM_FENCE)

#define threadIdx(x) get_local_id(0)
#define blockIdx(x) get_group_id(0)
#define blockDim(x) get_local_size(0)
#define gridDim(x) get_num_groups(0)

// All inlines must be static
#define inline static inline

// Silence standalone statics
#define static

#else

#define __global
#define __local
#define __constant __constant__

#define PRIVATE private :
#define PUBLIC public :

#endif // __OPENCL_VERSION__

#endif // DEVICE_CODE_H

