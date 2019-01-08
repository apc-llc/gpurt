#ifndef GPU_H
#define GPU_H

#if defined(WIN32)
#define __attribute__(...) __forceinline
#endif

#include "DeviceCode.h"

#if !defined(__OPENCL_VERSION__)

// Unify copy direction constants between CUDA and OpenCL.
typedef int memcpyKind;
extern const int memcpyHostToDevice;
extern const int memcpyDeviceToHost;

// Temporary turn off __AVX__ due to CUDA vs AVX512 incompatibility.
#ifdef __AVX__
#undef __AVX__
#include <CL/cl.h>
#define __AVX__
#else
#include <CL/cl.h>
#endif

union GPUerror_t
{
	cudaError_t cudaError;
	cl_int clError;
	int errcode;
};

#ifdef __cplusplus

#include <cstdarg>
#include <memory>
#include <string>
#include <vector>

// Packaging structure for kernel arguments.
struct KernelArgument
{
	void* addr;
	size_t size;
};

class GPU;

class IGPU
{
	virtual bool isAvailable() = 0;

	virtual const std::string& getPlatformName() = 0;
	
	virtual int getBlockSize() = 0;
	
	virtual int getBlockCount() = 0;
	
	virtual int getSMCount() = 0;
	
	virtual int getConstMemSize() = 0;
	
	virtual int getSharedMemSizePerSM() = 0;
	
	// Allocate global memory from the preallocated buffer.
	virtual void* malloc(size_t size) = 0;
	
	// Reset free memory pointer to the beginning of preallocated buffer.
	virtual void mfree() = 0;

	// Check whether the specified memory address belongs to GPU memory allocation.
	virtual bool isAllocatedOnGPU(const void* ptr) = 0;

	virtual GPUerror_t memset(void* dst, const int val, size_t size) = 0;

	virtual GPUerror_t memcpy(void* dst, const void* src, size_t size, memcpyKind kind) = 0;
	
	virtual GPUerror_t getLastError() = 0;

	virtual GPUerror_t launch(dim3 nblocks, dim3 szblock, unsigned int szshmem, void* stream,
		const char* name, const std::vector<KernelArgument>& kargs) = 0;

	virtual GPUerror_t synchronize() = 0;

	friend class GPU;
};

class GPU
{
	static std::unique_ptr<IGPU> gpu;

	GPU();

	static bool initGPU();

public :

	static bool isAvailable()
	{
		if (!initGPU())
			return false;

		return gpu->isAvailable();
	}

	static const std::string& getPlatformName()
	{
		static const std::string empty = "";
	
		if (!initGPU())
			return empty;

		return gpu->getPlatformName();
	}
	
	static int getBlockSize()
	{
		if (!initGPU())
			return 0;

		return gpu->getBlockSize();
	}
	
	static int getBlockCount()
	{
		if (!initGPU())
			return 0;

		return gpu->getBlockCount();
	}
	
	static int getSMCount()
	{
		if (!initGPU())
			return 0;

		return gpu->getSMCount();
	}
	
	static int getConstMemSize()
	{
		if (!initGPU())
			return 0;

		return gpu->getConstMemSize();
	}
	
	static int getSharedMemSizePerSM()
	{
		if (!initGPU())
			return 0;

		return gpu->getSharedMemSizePerSM();
	}
	
	// Allocate global memory from the preallocated buffer.
	static void* malloc(size_t size)
	{
		if (!initGPU())
			return NULL;

		return gpu->malloc(size);
	}
	
	// Reset free memory pointer to the beginning of preallocated buffer.
	static void mfree()
	{
		if (!initGPU())
			return;

		return gpu->mfree();
	}

	// Check whether the specified memory address belongs to GPU memory allocation.
	static bool isAllocatedOnGPU(const void* ptr)
	{
		if (!initGPU())
			return false;
		
		if (!ptr)
			return false;

		return gpu->isAllocatedOnGPU(ptr);
	}

	template<typename T>
	static GPUerror_t memset(T* dst, const int val, size_t size)
	{
		return gpu->memset(dst, val, size);
	}

	template<typename T>
	static GPUerror_t memcpy(T* dst, const T* src, size_t size, memcpyKind kind)
	{
		return gpu->memcpy(dst, src, size, kind);
	}
	
	static GPUerror_t getLastError()
	{
		return gpu->getLastError();
	}

	template<typename... Args>
	static GPUerror_t launch(dim3 nblocks, dim3 szblock, unsigned int szshmem, void* stream,
		const char* name, Args... args)
	{
		std::vector<void*> addrs = { &args... };
		std::vector<KernelArgument> kargs(addrs.size());
		for (int i = 0, e = addrs.size(); i != e; i++)
			kargs[i].addr = addrs[i]; 
		std::vector<size_t> sizes = { sizeof(args)... };
		for (int i = 0, e = sizes.size(); i != e; i++)
			kargs[i].size = sizes[i];
		
		return gpu->launch(nblocks, szblock, szshmem, stream, name, kargs);
	}

	static GPUerror_t synchronize()
	{
		return gpu->synchronize();
	}
};

#endif // __cplusplus

#endif // __OPENCL_VERSION__

#endif // GPU_H

