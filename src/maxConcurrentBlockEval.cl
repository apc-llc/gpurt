// The maximum number of registers the GPU kernels are expected to have.
// This value is used to calculate the maximum number of active ("persistent") blocks
// the target GPU can physically process in parallel without preemption.
// The application kernels shall be designed to launch this exact number of blocks, in order to
// process multiple loops in one kernel one by one and save time on synchronizations.
#define NREGS 32

#define NITS 10000

static inline void DelayFMADS(__global float* bigData)
{
	float values[NREGS];

	#pragma unroll
	for(int r = 0; r < NREGS; ++r)
		values[r] = bigData[get_local_id(0) + r * 32];

	#pragma unroll
	for(int i = 0; i < (NITS + NREGS - 1) / NREGS; ++i)
	{
		#pragma unroll
		for(int r = 0; r < NREGS; ++r)
			values[r] += values[r] * values[r];
		mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	}

	#pragma unroll
	for(int r = 0; r < NREGS; ++r)
		bigData[get_local_id(0) + r * 32] = values[r];
}

__kernel void maxConcurrentBlockEval(__global int* maxConcurrentBlocks,
	__global int* maxConcurrentBlockEvalDone, __global float* bigData)
{
	if (*maxConcurrentBlockEvalDone != 0)
		return;

	if (get_local_id(0) == 0)
		atomic_add(maxConcurrentBlocks, 1);

	DelayFMADS(bigData);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	*maxConcurrentBlockEvalDone = 1;
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

