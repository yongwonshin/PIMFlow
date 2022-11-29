#include <cuda.h>

__global__ void aggregation_kernel_cuda_start(void) {}
__global__ void aggregation_kernel_cuda_end(void) {}
__global__ void linear_kernel_cuda_start(void) {}
__global__ void linear_kernel_cuda_end(void) {}
__global__ void forward_kernel_cuda_start(void) {}
__global__ void forward_kernel_cuda_end(void) {}
__global__ void backward_kernel_cuda_start(void) {}
__global__ void backward_kernel_cuda_end(void) {}
__global__ void avgpool_kernel_cuda_start(void) {}
__global__ void avgpool_kernel_cuda_end(void) {}
__global__ void fc1_kernel_cuda_start(void) {}
__global__ void fc1_kernel_cuda_end(void) {}
__global__ void fc2_kernel_cuda_start(void) {}
__global__ void fc2_kernel_cuda_end(void) {}

void aggregation_kernel(bool is_start) {
  if (is_start)
    aggregation_kernel_cuda_start<<<1, 1>>>();
  else
    aggregation_kernel_cuda_end<<<1, 1>>>();

}
void linear_kernel(bool is_start) {
  if (is_start)
    linear_kernel_cuda_start<<<1, 1>>>();
  else
    linear_kernel_cuda_end<<<1, 1>>>();
}

void forward_kernel(bool is_start) {
  if (is_start)
    forward_kernel_cuda_start<<<1, 1>>>();
  else
    forward_kernel_cuda_end<<<1, 1>>>();
}

void backward_kernel(bool is_start) {
  if (is_start)
    backward_kernel_cuda_start<<<1, 1>>>();
  else
    backward_kernel_cuda_end<<<1, 1>>>();
}

void avgpool_kernel(bool is_start) {
  if (is_start)
    avgpool_kernel_cuda_start<<<1, 1>>>();
  else
    avgpool_kernel_cuda_end<<<1, 1>>>();
}

void fc1_kernel(bool is_start) {
  if (is_start)
    fc1_kernel_cuda_start<<<1, 1>>>();
  else
    fc1_kernel_cuda_end<<<1, 1>>>();
}

void fc2_kernel(bool is_start) {
  if (is_start)
    fc2_kernel_cuda_start<<<1, 1>>>();
  else
    fc2_kernel_cuda_end<<<1, 1>>>();
}
