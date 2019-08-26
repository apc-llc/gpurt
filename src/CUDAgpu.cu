#include "Check.h"
#include "CUDAgpu.h"

#include <cuda_runtime.h> // __cudaRegisterFunction
#ifndef WIN32
#include <dlfcn.h>
#else
#include <sstream>
#endif // WIN32
#include <iostream>
#include <map>
#include <memory>

// The maximum number of registers the GPU kernels are expected to have.
// This value is used to calculate the maximum number of active ("persistent") blocks
// the target GPU can physically process in parallel without preemption.
// The application kernels shall be designed to launch this exact number of blocks, in order to
// process multiple loops in one kernel one by one and save time on synchronizations.
#ifndef NREGS
#define NREGS 32
#endif

// Unify copy direction constants between CUDA and OpenCL.
const int memcpyHostToDevice = cudaMemcpyHostToDevice;
const int memcpyDeviceToHost = cudaMemcpyDeviceToHost;

using namespace std;

// Container that tracks correspondence between CUDA kernels
// pointers and their names. This is needed to unify kernel
// launch interfaces for CUDA and OpenCL.
unique_ptr<map<string, void*> > cuda_kernels;

// Track correspondence between CUDA kernels
// pointers and their names. This is needed to unify kernel
// launch interfaces for CUDA and OpenCL.
extern "C" void CUDARTAPI __cudaRegisterFunction(void** fatCubinHandle,
	const char* hostFun, char* deviceFun, const char* deviceName,
	int thread_limit, uint3* tid, uint3* bid, dim3* bDim, dim3* gDim, int* wSize)
{
	if (!cuda_kernels.get())
		cuda_kernels.reset(new map<string, void*>());

	(*cuda_kernels)[(string)deviceName] = (void*)hostFun;

	typedef cudaError_t (*__cudaRegisterFunction_t)(
		void**, const char*, char*, const char*,
		int, uint3*, uint3*, dim3*, dim3*, int*);
	static __cudaRegisterFunction_t __cudaRegisterFunction_;
	static int __cudaRegisterFunction_init = 0;
	if (!__cudaRegisterFunction_init)
	{
#if defined(WIN32)
		const static string cudartLibname = "cudart.dll";
		HMODULE hModule = LoadLibrary(cudartLibname.c_str());
		if (!hModule)
		{
			// Try alternative library naming cudartARCH_VERSION.dll.
			stringstream ss;
			ss << "cudart";

			bool is32bit = false;

			typedef BOOL(WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);

			LPFN_ISWOW64PROCESS fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(
				GetModuleHandle(TEXT("kernel32")), "IsWow64Process");

			// Use Wow64 as an indicator of 32-bit application running on 64-bit system.
			if (!fnIsWow64Process)
				is32bit = true;
			else
			{
				BOOL bIsWow64 = FALSE;
				if (!fnIsWow64Process(GetCurrentProcess(), &bIsWow64))
				{
					fprintf(stderr, "IsWow64Process() has failed\n");
					is32bit = false;
				}
				if (bIsWow64)
					is32bit = true;
			}

			ss << (is32bit ? "32" : "64");
			ss << "_";

			int runtimeVersion = 0;
			cudaRuntimeGetVersion(&runtimeVersion);

			ss << runtimeVersion / 1000; // major
			ss << (runtimeVersion % 1000) / 10;
			ss << ".dll";

			const string cudartLibnameAlt = ss.str();

			hModule = LoadLibrary(cudartLibnameAlt.c_str());
			if (!hModule)
			{
				fprintf(stderr, "Error loading CUDA runtime library \"%s\" or \"%s\"\n",
					cudartLibname.c_str(), cudartLibnameAlt.c_str());
				exit(-1);
			}
		}
		__cudaRegisterFunction_ = (__cudaRegisterFunction_t)GetProcAddress(hModule, "__cudaRegisterFunction");
		if (!__cudaRegisterFunction_)
		{
			fprintf(stderr, "Erro loading function \"%s\" during CUDA runtime startup\n", "__cudaRegisterFunction");
			exit(-1);
		}
#else
		void* handle = dlopen("libcudart.so", RTLD_LAZY);
		if (!handle)
		{
			fprintf(stderr, "Error loading CUDA runtime library \"libcudart.so\": %s\n", dlerror());
			exit(-1);
		}
		__cudaRegisterFunction_ = (__cudaRegisterFunction_t)dlsym(handle, "__cudaRegisterFunction");
		if (!__cudaRegisterFunction_)
		{
			fprintf(stderr, "Erro loading function \"%s\" during CUDA runtime startup\n", "__cudaRegisterFunction");
			exit(-1);
		}
#endif // WIN32
		__cudaRegisterFunction_init = 1;
	}
	if (__cudaRegisterFunction_)
		__cudaRegisterFunction_(fatCubinHandle, hostFun, deviceFun, deviceName,
			thread_limit, tid, bid, bDim, gDim, wSize);
}

template<int ITS, int REGS>
inline __attribute__((always_inline))
__device__
static void DelayFMADS(float* bigData)
{
	float values[REGS];

	#pragma unroll
	for(int r = 0; r < REGS; ++r)
		values[r] = bigData[threadIdx.x + r * 32];

	#pragma unroll
	for(int i = 0; i < (ITS + REGS - 1) / REGS; ++i)
	{
		#pragma unroll
		for(int r = 0; r < REGS; ++r)
			values[r] += values[r] * values[r];
		__threadfence_block();
	}

	#pragma unroll
	for(int r = 0; r < REGS; ++r)
		bigData[threadIdx.x + r * 32] = values[r];
}

__global__
void maxConcurrentBlockEval(int* maxConcurrentBlocks, int* maxConcurrentBlockEvalDone, float* bigData)
{
	if (*maxConcurrentBlockEvalDone != 0)
		return;

	if (threadIdx.x == 0)
		atomicAdd(maxConcurrentBlocks, 1);

	DelayFMADS<10000, NREGS>(bigData);
	__syncthreads();

	*maxConcurrentBlockEvalDone = 1;
	__threadfence();
}

bool CUDAgpu::initGPU()
{
	if (fatalError != cudaSuccess)
		return false;
	
	return true;
}

bool CUDAgpu::isAvailable()
{
	if (!initGPU())
		return false;

	return (ngpus > 0);
}

const string& CUDAgpu::getPlatformName()
{
	static const string platformName = "CUDA";
	return platformName;
}

int CUDAgpu::getCC()
{
	if (!initGPU())
		return 0;
	
	return cc;
}

int CUDAgpu::getBlockSize()
{
	return 128;
}

int CUDAgpu::getBlockCount()
{
	if (!initGPU())
		return 0;

	return nblocks;
}

int CUDAgpu::getSMCount()
{
	if (!initGPU())
		return 0;

	return nsms;
}

int CUDAgpu::getConstMemSize()
{
	if (!initGPU())
		return 0;

	return szcmem;
}

int CUDAgpu::getSharedMemSizePerSM()
{
	if (!initGPU())
		return 0;

	return szshmem;
}

int CUDAgpu::getSharedMemSizePerBlock()
{
	if (!initGPU())
		return 0;

	return szshmemPerBlock;
}

void* CUDAgpu::malloc(size_t size)
{
#define MALLOC_ALIGNMENT 256

	if (!initGPU())
		return NULL;

	if (!gmem) return NULL;

	if (ptr + size + MALLOC_ALIGNMENT > gmem + szgmem)
		return NULL;
	
	void* result = ptr;
	ptr += size;
	
	ptrdiff_t alignment = (ptrdiff_t)ptr % MALLOC_ALIGNMENT;
	if (alignment)
		ptr += MALLOC_ALIGNMENT - alignment;
	
	return result;
}

// Reset free memory pointer to the beginning of preallocated buffer.
void CUDAgpu::mfree()
{
	if (!initGPU())
		return;

	ptr = gmem;
}

// Check whether the specified memory address belongs to GPU memory allocation.
bool CUDAgpu::isAllocatedOnGPU(const void* ptr)
{
	if (!initGPU())
		return false;
	
	if (!gmem) return false;

	if ((ptr >= gmem) && (ptr <= gmem + szgmem))
		return true;
	
	return false;
}

GPUerror_t CUDAgpu::memset(void* dst, const int val, size_t size)
{
	if (!initGPU())
		return { cudaErrorNoDevice };
	
	cudaError_t cudaError;
	CUDA_ERR_CHECK(cudaError = cudaMemset(dst, val, size));
	if (cudaError != cudaSuccess)
	{
		fatalError = cudaError;
		return { fatalError };
	}
	
	return { cudaError };
}

GPUerror_t CUDAgpu::memcpy(void* dst, const void* src, size_t size, memcpyKind kind)
{
	if (!initGPU())
		return { cudaErrorNoDevice };
	
	cudaError_t cudaError;
	CUDA_ERR_CHECK(cudaError = cudaMemcpy(dst, src, size, (cudaMemcpyKind)kind));
	if (cudaError != cudaSuccess)
	{
		fatalError = cudaError;
		return { fatalError };
	}
	
	return { cudaError };
}

GPUerror_t CUDAgpu::getLastError()
{
	// If GPU is not initialized, then there is either no
	// device or fatal error during initialization.
	if (!initGPU())
		return { fatalError };

	return { cudaGetLastError() };
}

GPUerror_t CUDAgpu::launch(dim3 nblocks, dim3 szblock, unsigned int szshmem, void* stream,
	const char* name, const std::vector<KernelArgument>& kargs_)
{
	if (!initGPU())
		return { cudaErrorNoDevice };

	// Make sure the kernels index exists.
	if (!cuda_kernels.get())
		return { cudaErrorNotReady };

	// Unlike for OpenCL, for CUDA we only need arguments addresses, without sizes.
	vector<void*> kargs(kargs_.size() + 2);
	for (int i = 0, e = kargs_.size(); i < e; i++)
		kargs[i] = kargs_[i].addr;

	// The kernel must be indexed either in precompiled or
	// JIT-compiled kernels index.
	void* shmem = NULL;
	void* kernel = NULL;
	if (cuda_kernels->find((string)name) != cuda_kernels->end())
	{		
		// Add two extra artificial arguments that mostly make sense for OpenCL:
		// local memory pointer and local memory size.
		kargs[kargs_.size()] = &shmem;
		kargs[kargs_.size() + 1] = &szshmem;

		// Launch precompiled CUDA kernel.
		kernel = (*cuda_kernels)[(string)name];
	}
	else if (cuda_kernels_jit.find((string)name) != cuda_kernels_jit.end())
	{
		// Add two extra artificial arguments that mostly make sense for OpenCL:
		// local memory pointer and local memory size. Note here we artificially
		// send global memory pointer instead of shared memory.
		kargs[kargs_.size()] = &shmem_debug;
		kargs[kargs_.size() + 1] = &szshmem;

		// Launch JIT-compiled CUDA kernel.
		kernel = cuda_kernels_jit[(string)name];
	}
	else
		return { cudaErrorInvalidDeviceFunction };

	cudaError_t cudaError;
	CUDA_ERR_CHECK(cudaError = cudaLaunchCooperativeKernel(
		kernel, nblocks, szblock, &kargs[0], szshmem, stream ? *(cudaStream_t*)stream : 0));
	if (cudaError != cudaSuccess)
	{
		fatalError = cudaError;
		return { fatalError };
	}

	return { fatalError };
}

GPUerror_t CUDAgpu::launch(dim3 nblocks, dim3 szblock, unsigned int szshmem, void* stream,
	const char* name_, const char* ptx_, const std::vector<KernelArgument>& kargs_)
{
	if (!initGPU())
		return { cudaErrorNoDevice };

	string name = (string)name_ + "_opencl_debug";

	// Check if the kernel name is indexed.
	if (cuda_kernels_jit.find(name) == cuda_kernels_jit.end())
	{
		CUresult cuResult = CUDA_SUCCESS;

		string ptx = ptx_;

		// Replace .shared with .global, as the CUDA context
		// cannot work with .shared parameter, which is used
		// in OpenCL to pass shared memory region through
		// the kernel arguments (CUDA_ERROR_INVALID_IMAGE = 200).
		string search = ".shared", replace = ".global";
		size_t pos = 0;
		while ((pos = ptx.find(search, pos)) != string::npos)
		{
			ptx.replace(pos, search.length(), replace);
			pos += replace.length();
		}

		// Replace name with name_opencl_debug.
		search = string(".entry ") + name_, replace = string(".entry ") + name;
		pos = 0;
		while ((pos = ptx.find(search, pos)) != string::npos)
		{
			ptx.replace(pos, search.length(), replace);
			pos += replace.length();
		}

		CUjit_option options[] =
		{
			CU_JIT_TARGET,
			CU_JIT_MAX_REGISTERS,
#if !defined(NDEBUG)
			CU_JIT_INFO_LOG_BUFFER,
			CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
			CU_JIT_ERROR_LOG_BUFFER,
			CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
			CU_JIT_OPTIMIZATION_LEVEL,
			CU_JIT_GENERATE_DEBUG_INFO,
			CU_JIT_LOG_VERBOSE,
#endif // NDEBUG
		};

		CUmodule module;
		{
			const size_t szstdout = 1024 * 1024;
			vector<char> vstdout(szstdout);

			const size_t szstderr = 1024 * 1024;
			vector<char> vstderr(szstderr);

			void *optionValues[] =
			{
				(void*)(uintptr_t)cc,
				(void*)(uintptr_t)NREGS,
#if !defined(NDEBUG)
				(void*)(uintptr_t)&vstdout[0],
				(void*)(uintptr_t)szstdout,
				(void*)(uintptr_t)&vstderr[0],
				(void*)(uintptr_t)szstderr,
				(void*)(uintptr_t)0,
				(void*)(uintptr_t)1,
				(void*)(uintptr_t)1,
#endif // NDEBUG
			}; 

			// JIT-compile the given PTX source.
			cuResult = cuModuleLoadDataEx(&module, ptx.c_str(),
				sizeof(options) / sizeof(options[0]), options, optionValues);
		
			// Output logs.
			for (size_t i = 0; i < szstdout; i++)
				if (vstdout[i] == '\0')
				{
					cout << (char*)&vstdout[0] << endl;
					break;
				}
			for (size_t i = 0; i < szstderr; i++)
				if (vstderr[i] == '\0')
				{
					cerr << (char*)&vstderr[0] << endl;
					break;
				}
		}
#if !defined(NDEBUG)	
		CU_ERR_CHECK(cuResult);
#endif // NDEBUG
		if (cuResult != CUDA_SUCCESS)
		{
			fatalError = cudaErrorInvalidKernelImage;
			return { fatalError };
		}

		CUfunction function;
		CU_ERR_CHECK(cuResult = cuModuleGetFunction(&function, module, name.c_str()));
		if (cuResult != CUDA_SUCCESS)
		{
			fatalError = cudaErrorInvalidKernelImage;
			return { fatalError };
		}
	
		// Store the loaded function to the index of JIT-ted functions.
		cuda_kernels_jit[name] = function;
	}
	
	// Chain to generic launcher.
	return launch(nblocks, szblock, szshmem, stream, name.c_str(), kargs_);
}

GPUerror_t CUDAgpu::synchronize()
{
	if (!initGPU())
		return { cudaErrorNoDevice };

	return { cudaDeviceSynchronize() };
}

const char* CUDAgpu::getErrorString(int code)
{
	return cudaGetErrorString((cudaError_t)code);
}

CUDAgpu::CUDAgpu() : fatalError(cudaSuccess), ngpus(0), gmem(NULL), ptr(NULL)
{
	cudaError_t cudaError;

#define CUDA_RETURN_ON_ERR(x) do { \
	CUDA_ERR_CHECK(x); if (cudaError != cudaSuccess) { fatalError = cudaError; return; } \
} while (0)
	
	CUDA_ERR_CHECK(cudaError = cudaGetDeviceCount(&ngpus));
	if ((cudaError != cudaSuccess) && (cudaError != cudaErrorNoDevice))
	{
		fatalError = cudaError;
		return;
	}

	if (!ngpus) return;

	int* maxConcurrentBlocks = NULL;
	CUDA_RETURN_ON_ERR(cudaError = cudaMalloc(&maxConcurrentBlocks, sizeof(int)));
	CUDA_RETURN_ON_ERR(cudaError = cudaMemset(maxConcurrentBlocks, 0, sizeof(int)));
	
	int* maxConcurrentBlockEvalDone = NULL;
	CUDA_RETURN_ON_ERR(cudaError = cudaMalloc(&maxConcurrentBlockEvalDone, sizeof(int)));
	CUDA_RETURN_ON_ERR(cudaError = cudaMemset(maxConcurrentBlockEvalDone, 0, sizeof(int)));

	float* bigData = NULL;
	CUDA_RETURN_ON_ERR(cudaError = cudaMalloc(&bigData, 1024 * 1024 * sizeof(float)));

	maxConcurrentBlockEval<<<1024, getBlockSize()>>>(maxConcurrentBlocks, maxConcurrentBlockEvalDone, bigData);

	CUDA_RETURN_ON_ERR(cudaError = cudaGetLastError());

	CUDA_RETURN_ON_ERR(cudaError = cudaDeviceSynchronize());

	CUDA_RETURN_ON_ERR(cudaError = cudaMemcpy(&nblocks, maxConcurrentBlocks, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_RETURN_ON_ERR(cudaError = cudaFree(maxConcurrentBlocks));

	CUDA_RETURN_ON_ERR(cudaError = cudaFree(maxConcurrentBlockEvalDone));

	CUDA_RETURN_ON_ERR(cudaError = cudaFree(bigData));
		
	cudaDeviceProp props;
	CUDA_RETURN_ON_ERR(cudaError = cudaGetDeviceProperties(&props, 0));

	cc = props.major * 10 + props.minor;

	nsms = props.multiProcessorCount;

	szcmem = props.totalConstMem;
	
	szshmem = props.sharedMemPerMultiprocessor;
	
	szshmemPerBlock = min(props.sharedMemPerBlock, (nsms * szshmem) / nblocks);

	std::cout << "Using GPU " << props.name << " : max concurrent blocks = " << nblocks <<
		" : " << szshmemPerBlock << "B of shmem per block" << std::endl;

	// Preallocate 85% of GPU memory to save on costly subsequent allocations.
	size_t available, total;
	CUDA_RETURN_ON_ERR(cudaError = cudaMemGetInfo(&available, &total));

	szgmem = (size_t)(0.85 * available);

	CUDA_RETURN_ON_ERR(cudaError = cudaMalloc(&gmem, szgmem));

	ptr = gmem;

	// Allocate global memory scratch space for emulating
	// shared memory in OpenCL kernels executed through the
	// CUDA pipeline.
	shmem_debug = (char*)this->malloc(szshmemPerBlock * nblocks);
}

CUDAgpu::~CUDAgpu()
{
	cudaFree(gmem);
}

