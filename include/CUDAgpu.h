#ifndef CUDA_GPU_H
#define CUDA_GPU_H

#include "GPU.h"

#include <cuda.h>
#include <map>
#include <string>

#if !defined(NDEBUG)
class CLgpu;
#endif

class CUDAgpu : public IGPU
{
	// If not cudaSuccess, indicates that the CUDAgpu::* functions are not usable at all.
	// (checked by every function call in the beginning).
	cudaError_t fatalError;

	// Container for runtime-compiled CUDA kernels.
	std::map<std::string, CUfunction> cuda_kernels_jit;

	int ngpus;
	int cc;
	int nblocks;
	int nsms;
	size_t szcmem;
	size_t szshmem;
	size_t szgmem;
	
	char *gmem, *ptr;

	// Allocate global memory scratch space for emulating
	// shared memory in OpenCL kernels executed through the
	// CUDA pipeline.	
	char *shmem_debug;

	bool initGPU();

	virtual bool isAvailable();
	
	virtual const std::string& getPlatformName();
	
	virtual int getCC();
	
	virtual int getBlockSize();
	
	virtual int getBlockCount();
	
	virtual int getSMCount();
	
	virtual int getConstMemSize();
	
	virtual int getSharedMemSizePerSM();
	
	// Allocate global memory from the preallocated buffer.
	virtual void* malloc(size_t size);
	
	// Reset free memory pointer to the beginning of preallocated buffer.
	virtual void mfree();

	// Check whether the specified memory address belongs to GPU memory allocation.
	virtual bool isAllocatedOnGPU(const void* ptr);

	virtual GPUerror_t memset(void* dst, const int val, size_t size);

	virtual GPUerror_t memcpy(void* dst, const void* src, size_t size, memcpyKind kind);
	
	virtual GPUerror_t getLastError();

	virtual GPUerror_t launch(dim3 nblocks, dim3 szblock, unsigned int szshmem, void* stream,
		const char* name, const std::vector<KernelArgument>& kargs);

	virtual GPUerror_t launch(dim3 nblocks, dim3 szblock, unsigned int szshmem, void* stream,
		const char* name, const char* ptx, const std::vector<KernelArgument>& kargs);

	virtual GPUerror_t synchronize();
	
	CUDAgpu();

public :

	static const char* getErrorString(int code);

	virtual ~CUDAgpu();

	friend class GPU;

#if !defined(NDEBUG)
	friend class CLgpu;
#endif
};

#endif // CUDA_GPU_H

