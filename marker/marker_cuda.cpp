#include <torch/extension.h>
//#include <cuda.h>

void aggregation_kernel(bool is_start);
void linear_kernel(bool is_start);
void forward_kernel(bool is_start);
void backward_kernel(bool is_start);
void avgpool_kernel(bool is_start);
void fc1_kernel(bool is_start);
void fc2_kernel(bool is_start);

void aggregation(bool is_start) {
  aggregation_kernel(is_start);
}
void linear(bool is_start) {
  linear_kernel(is_start);
}

void forward(bool is_start) {
  forward_kernel(is_start);
}
void backward(bool is_start) {
  backward_kernel(is_start);
}

void avgpool(bool is_start) {
  avgpool_kernel(is_start);
}
void fc1(bool is_start) {
  fc1_kernel(is_start);
}
void fc2(bool is_start) {
  fc2_kernel(is_start);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aggregation", &aggregation, "GCN aggregation");
  m.def("linear", &linear, "GCN linear");
  m.def("forward", &forward, "forward");
  m.def("backward", &backward, "backward");
  m.def("avgpool", &avgpool, "avgpool");
  m.def("fc1", &fc1, "fc1");
  m.def("fc2", &fc2, "fc2");
}
