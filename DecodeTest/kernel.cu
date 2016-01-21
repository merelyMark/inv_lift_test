
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

template<class Int>
__host__ __device__
static void
inv_lift_orig(Int* p, unsigned int s)
{
	Int x, y, z, w;
	x = *p; p += s;
	y = *p; p += s;
	z = *p; p += s;
	w = *p; p += s;

	y += w >> 1; w -= y >> 1;
	y += w; w <<= 1; w -= y;
	z += x; x <<= 1; x -= z;
	y += z; z <<= 1; z -= y;
	w += x; x <<= 1; x -= w;

	p -= s; *p = w;
	p -= s; *p = z;
	p -= s; *p = y;
	p -= s; *p = x;
}
template<class Int>
__host__ __device__
static void
inv_lift_fix2(Int* p, unsigned int s)
{
	volatile Int x, y, z, w;
	x = *p; p += s;
	y = *p; p += s;
	z = *p; p += s;
	w = *p; p += s;

	y += w >> 1; w -= y >> 1;
	y += w; w <<= 1; w -= y;
	z += x; x <<= 1; x -= z;
	y += z; z <<= 1; z -= y;
	w += x; x <<= 1; x -= w;

	p -= s; *p = w;
	p -= s; *p = z;
	p -= s; *p = y;
	p -= s; *p = x;
}
template<class Int>
__host__ __device__
static void
inv_lift_fix1(Int* p, unsigned int s)
{
	Int x, y, z, w;
	x = *p;
	y = p[s * 1];
	z = p[s * 2];
	w = p[s * 3];

	y += w >> 1; w -= y >> 1;
	y += w; w <<= 1; w -= y;
	z += x; x <<= 1; x -= z;
	y += z; z <<= 1; z -= y;
	w += x; x <<= 1; x -= w;

	p[s * 3] -= s;
	p[s * 2] -= s;
	p[s] -= s;
	p[0] -= s;
}



template<class Int>
__global__
void gpuTest
(
Int *iblock
)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;
	inv_lift_orig(iblock + idx * 64, 16);


}

template<class Int>
void cpuTest(std::vector<Int> &h_c)
{
	int i = 0;
	//for (i = 0; i < h_c.size(); i++){
	inv_lift_orig(thrust::raw_pointer_cast(h_c.data()) + i * 64, 16);
	//}

}
typedef long long Int;

int main()
{
	const int nx = 256;
	const int ny = 256;
	const int nz = 256;
    const int arraySize = nx*ny*nz;
	const int thread_cnt = arraySize / 64;
	thrust::host_vector<Int> h_cout, h_a;
	h_cout.resize(arraySize);
	h_a.resize(arraySize);

	for (int i = 0; i < arraySize; i++){
		h_a[i] = i;
	}

	thrust::device_vector<Int> d_a, d_c;
	d_a.resize(arraySize);
	d_c.resize(arraySize);

	d_a = h_a;
	d_c = h_a;
    

	gpuTest<Int> << <1,1 >> >(thrust::raw_pointer_cast(d_c.data()));
	//dim3 emax_size(nx / 4, ny / 4, nz / 4);
	//dim3 block_size(8, 8, 8);
	//dim3 grid_size = emax_size;
	//grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;
	//cudaInvXForm<Int> << <block_size, grid_size >> >(thrust::raw_pointer_cast(d_c.data()));
	//cudaStreamSynchronize(0);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bitshiftKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(1);
	}

	h_cout = d_c;

	std::vector<Int> h_c;
	h_c.resize(arraySize);
	thrust::copy(h_a.begin(), h_a.end(), h_c.begin());
	cpuTest<Int>(h_c);

	for (int i = 0; i < h_c.size(); i++){
		if (h_c[i] != h_cout[i]){
			std::cout << i << " " << h_c[i] << " " << h_cout[i] << std::endl;
			exit(1);
		}
	}

    return 0;
}

