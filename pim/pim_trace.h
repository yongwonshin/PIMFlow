#include <iostream>
#include <fstream>
#include <sstream>

namespace pim {
typedef struct strideInfo {
  bool use_stride = false;
  int num_first_elem = 0;
  int stride = 0;
  int num_after_elem = 0;
  int num_gwrite = 1;
} StrideInfo;
std::string ToBinary(int n);
std::string FillZero(std::string s, size_t n);

void GWrite(std::ostream& OS, char* buf, int r, StrideInfo sinfo=StrideInfo());
void GAct(std::ostream& OS, char* buf, int k, int r, int j, int num_act);
void Comp(std::ostream& OS, char* buf, int k, int r, int j, int h);

void ReadRes(std::ostream& OS);

void OutputNewtonTrace(std::ostream& OS, std::string kernel_name, int64_t row, int64_t col);
void OutputNewtonTraceV2(std::ostream& OS, std::string kernel_name, int64_t row, int64_t col, int64_t stride, StrideInfo sinfo=StrideInfo());
}
