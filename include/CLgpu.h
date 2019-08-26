#ifndef CL_GPU_H
#define CL_GPU_H

#include "GPU.h"

#include <map>
#include <string>
#include <vector>

#if !defined(NDEBUG)
class CUDAgpu;
#endif

class CLgpu : public IGPU
{
	// If not CL_SUCCESS, indicates that the CLgpu::* functions are not usable at all.
	// (checked by every function call in the beginning).
	cl_int fatalError;

	// Container for runtime-compiled OpenCL kernels.
	std::map<std::string, cl_kernel> cl_kernels;

	// Container for compiled IR for embedded OpenCL sources
	// that shall be used for debugging on NVIDIA.
	std::map<std::string, std::string> cl_binaries;

	cl_context context;
	cl_device_id device;
	std::vector<cl_program> programs;
	cl_command_queue cmdQueue;

	unsigned int ngpus;
	int nblocks;
	int nsms;
	int szcmem;
	int szshmem, szshmemPerBlock;
	size_t szgmem;
	
	char *gmem, *ptr;
	cl_mem gmem_cl;

	bool initGPU();

	virtual bool isAvailable();

	virtual const std::string& getPlatformName();
	
	virtual int getBlockSize();
	
	virtual int getBlockCount();
	
	virtual int getSMCount();
	
	virtual int getConstMemSize();
	
	virtual int getSharedMemSizePerSM();

	virtual int getSharedMemSizePerBlock();
	
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

	virtual GPUerror_t synchronize();

	CLgpu();

#if !defined(NDEBUG)
	std::unique_ptr<CUDAgpu> cudaGPU;
#endif

public :

	static const char* getErrorString(cl_int code);

	virtual ~CLgpu();

	friend class GPU;
};

#endif // CL_GPU_H

