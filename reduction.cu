#include <iostream>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    std::cout << "GPU assert failed" << std::endl;
    if (abort) exit(code);
  }
}

__global__ void reduction_kernel0(int *d_out, int *d_in, int size) {
  constexpr int block_size = 256;
  __shared__ int sdata[block_size];
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto tid = threadIdx.x;
  if (i < size) sdata[tid] = d_in[i];
  else sdata[tid] = 0;
  __syncthreads();

  for (int step = 1; tid+step < block_size; step *= 2) {
    if (tid % (2 * step) == 0) {
      sdata[tid] += sdata[tid + step];
    }
    __syncthreads();
  }
  if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

__global__ void reduction_kernel1(int *d_out, int *d_in, int size) {
  constexpr int block_size = 256;
  __shared__ int sdata[block_size];
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto tid = threadIdx.x;
  if (i < size) sdata[tid] = d_in[i];
  else sdata[tid] = 0;
  __syncthreads();

  for (int step = 1; tid+step < block_size; step *= 2) {
    auto index = 2*step*tid;
    if (index < block_size) {
      sdata[index] += sdata[index + step];
    }
    __syncthreads();
  }
  if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

__global__ void reduction_kernel2(int *d_out, int *d_in, int size) {
  constexpr int block_size = 256;
  __shared__ int sdata[block_size];
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto tid = threadIdx.x;
  if (i < size) sdata[tid] = d_in[i];
  else sdata[tid] = 0;
  __syncthreads();

  for (int step = blockDim.x/2; step > 0; step /= 2) {
    if (tid < step) {
      sdata[tid] += sdata[tid + step];
    }
    __syncthreads();
  }
  if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

__global__ void reduction_kernel3(int *d_out, int *d_in, int size) {
  constexpr int block_size = 256;
  __shared__ int sdata[block_size];
  auto tid = threadIdx.x;
  auto i = blockIdx.x * (2*blockDim.x) + threadIdx.x;
  if (i+blockDim.x < size)
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
  else if (i < size) sdata[tid] = d_in[i];
  else sdata[tid] = 0;
  __syncthreads();

  for (int step = blockDim.x/2; step > 0; step /= 2) {
    if (tid < step) {
      sdata[tid] += sdata[tid + step];
    }
    __syncthreads();
  }
  if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

__device__ void warpReduce(int *sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void reduction_kernel4(int *d_out, int *d_in, int size) {
  constexpr int block_size = 256;
  constexpr int warp_size = 32;
  __shared__ int sdata[block_size];
  auto tid = threadIdx.x;
  auto i = blockIdx.x * (2*blockDim.x) + threadIdx.x;
  if (i+blockDim.x < size)
    sdata[tid] = d_in[i] + d_in[i + blockDim.x];
  else if (i < size) sdata[tid] = d_in[i];
  else sdata[tid] = 0;
  __syncthreads();

  // NOTE: step > warp_size (was step > 1)
  for (int step = blockDim.x/2; step > warp_size; step /= 2) {
    if (tid < step) {
      sdata[tid] += sdata[tid + step];
    }
    __syncthreads();
  }
  // no ifs in last warp
  if (tid < warp_size) warpReduce(sdata, tid);
  if (tid == 0) d_out[blockIdx.x] = sdata[0];
}



auto timeit(std::string const & name, int nrepeat, auto && worker)
{
  using namespace std::chrono;
  std::cout << "Running \'" << name << "\'" << std::endl;
  auto const start_time = high_resolution_clock::now();
  for (int i = 0; i < nrepeat; ++i) {
    if (nrepeat != 1 && nrepeat < 10)
      std::cout << "Repeat " << i + 1 << "/" << nrepeat << std::endl;
    worker();
  }

  auto const end_time = high_resolution_clock::now();
  auto const duration = (duration_cast<microseconds>(end_time - start_time)).count();
  std::cout << "Test \'" << name << "\'"
            << " took " << (double)duration / (double)nrepeat << " [us]";
  if (nrepeat != 1) std::cout << " (average)";
  std::cout << std::endl;
  return duration / (double)nrepeat;
}

void compare(std::string name,
             thrust::device_vector<int> const& sol,
             thrust::device_vector<int> const& x,
             thrust::device_vector<int>& y_test,
             auto && worker) {
  thrust::fill(y_test.begin(), y_test.end(), 0);
  worker();
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  if (sol != y_test) {
    std::cout << "Test " << name <<" failed" << std::endl;
    exit(1);
  }
}

auto main(int argc, char *argv[]) -> int {

  // correctness
  {
    int threads_per_block = 256;
    int data_size = 1 << 10;
    thrust::device_vector<int> x(data_size);
    thrust::sequence(x.begin(), x.end());
    thrust::device_vector<int> y_true(data_size);
    thrust::fill(y_true.begin(), y_true.end(), 0);
    thrust::device_vector<int> y_test(data_size);
    thrust::fill(y_test.begin(), y_test.end(), 0);
    int n_blocks = (x.size() + threads_per_block - 1) / threads_per_block;

    reduction_kernel0<<<n_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(y_true.data()),
        thrust::raw_pointer_cast(x.data()), x.size());
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    compare("test1", y_true, x, y_test, [&] {
      reduction_kernel1<<<n_blocks, threads_per_block>>>(
          thrust::raw_pointer_cast(y_test.data()),
          thrust::raw_pointer_cast(x.data()), x.size());
    });
    compare("test2", y_true, x, y_test, [&] {
    reduction_kernel2<<<n_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(y_test.data()),
        thrust::raw_pointer_cast(x.data()), x.size());
    });
    // compare("test3", y_true, x, y_test, [&] {
    //     std::cout << "running test 3 " << "nb = " << n_blocks << " nb/2 = " << n_blocks/2 << std::endl;

    // reduction_kernel3<<<n_blocks/2, threads_per_block>>>(
    //     thrust::raw_pointer_cast(y_test.data()),
    //     thrust::raw_pointer_cast(x.data()), x.size());
    // });
  }

  thrust::device_vector<int> x(1 << 24);
  thrust::sequence(x.begin(), x.end());
  thrust::device_vector<int> y(1 << 24);
  int threads_per_block = 256;
  int n_blocks = (x.size() + threads_per_block - 1) / threads_per_block;
  int n_repeat = 1000;
  timeit("reduction0", n_repeat, [&] {
      reduction_kernel0<<<n_blocks, threads_per_block>>>(thrust::raw_pointer_cast(y.data()),
                                                        thrust::raw_pointer_cast(x.data()), x.size());
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
  });
  timeit("reduction1", n_repeat, [&] {
      reduction_kernel1<<<n_blocks, threads_per_block>>>(thrust::raw_pointer_cast(y.data()),
                                                        thrust::raw_pointer_cast(x.data()), x.size());
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
  });
  timeit("reduction2", n_repeat, [&] {
      reduction_kernel2<<<n_blocks, threads_per_block>>>(thrust::raw_pointer_cast(y.data()),
                                                        thrust::raw_pointer_cast(x.data()), x.size());
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
  });
  timeit("reduction3", n_repeat, [&] {
      reduction_kernel3<<<n_blocks/2, threads_per_block>>>(thrust::raw_pointer_cast(y.data()),
                                                           thrust::raw_pointer_cast(x.data()), x.size());
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
  });
  timeit("reduction4", n_repeat, [&] {
      reduction_kernel4<<<n_blocks/2, threads_per_block>>>(thrust::raw_pointer_cast(y.data()),
                                                           thrust::raw_pointer_cast(x.data()), x.size());
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
  });

  return 0;
}
