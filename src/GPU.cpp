#include "Check.h"
#include "GPU.h"
#include "CUDAgpu.h"
#include "CLgpu.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

using namespace std;

static mutex gpuMutex;

bool GPU::initGPU()
{
	bool result = true;

	// First look if we are required to boot specifically CUDA or OpenCL.
	// If this is not the case, try to boot NVIDIA CUDA GPU first.
	// If CUDA device is not available, try to boot OpenCL device.
	if (!gpu.get())
	{
		gpuMutex.lock();
		while (!gpu.get())
		{
			char* cuse_cuda = getenv("USE_CUDA");
			int use_cuda = -1;
			if (cuse_cuda)
			{
				if (stringstream(cuse_cuda) >> use_cuda)
				{
					if (use_cuda == 1)
					{
						gpu.reset(new CUDAgpu());
						break;
					}
				}
			}
			
			char* cuse_opencl = getenv("USE_OPENCL");
			int use_opencl = -1;
			if (cuse_opencl)
			{
				if (stringstream(cuse_opencl) >> use_opencl)
				{
					if (use_opencl == 1)
					{
						gpu.reset(new CLgpu());
						break;
					}
				}
			}
			
			if ((use_cuda == 0) && (use_opencl == 0))
			{
				result = false;
				break;
			}
		
			gpu.reset(new CUDAgpu());
			if (!gpu->isAvailable())
			{
				gpu.reset(new CLgpu());
				if (!gpu->isAvailable())
					result = false;
			}
			break;
		}
		gpuMutex.unlock();
	}

	return result;
}

unique_ptr<IGPU> GPU::gpu;

