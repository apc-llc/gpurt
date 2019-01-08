__kernel void getRawPointer(__global void* ptr, __global ptrdiff_t* result)
{
	*result = (ptrdiff_t)ptr;
}

