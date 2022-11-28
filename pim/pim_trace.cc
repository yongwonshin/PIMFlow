#include "pim_trace.h"
#include <iomanip>
#include <cstring>
#include <sstream>

namespace pim {
std::string ToBinary(int n) {
  static char b[15]; // TB
  static char r[15];
  b[0] = '\0';
  while (n > 0) {
    strcat(b, ((n % 2) == 0) ? "0" : "1");
    n /= 2;
  }
  for (size_t i = 0; i < strlen(b); i++)
    r[strlen(b)-1-i] = b[i];
  r[strlen(b)] = '\0';
  return r;
}

std::string FillZero(std::string s, size_t n) {
  if (s.size() > 100) std::invalid_argument("string size too large!");

  static char buf[100]; // arbitrary size large enough
  for (size_t i = 0; i < n; i++) {
    if (i < s.size()) buf[n-s.size()+i] = s[i];
    else buf[n-1-i] = '0';
  }
  buf[n] = '\0';
  return buf;
}

void GWrite(std::ostream& OS, char* buf, int r, StrideInfo sinfo) {
  sprintf(buf, "%02d%02d%s%05d%06d", 0, 0, FillZero(ToBinary(r), 14).c_str(), 0, 0);
  OS << "0x" << std::setfill('0') << std::setw(8) << std::hex << std::stoi(buf, nullptr, 2) << " GWRITE ";
  if (sinfo.use_stride) {
    OS << std::dec << sinfo.num_first_elem << " " << sinfo.stride << " " << sinfo.num_after_elem;
  }
  OS << "\n";
  OS.flush();
}

void GAct(std::ostream& OS, char* buf, int k, int r, int j, int num_act) {
  for (int i = 0; i < num_act; i++) {
    sprintf(buf, "%s%02d%s%05d%06d", FillZero(ToBinary(i), 2).c_str(), 0, FillZero(ToBinary(k*r+j), 14).c_str(), 0, 0);
    OS << "0x" << std::setfill('0') << std::setw(8) << std::hex << std::stoi(buf, nullptr, 2) << " G_ACT" << i << "\n";
  }
  OS.flush();
}

void Comp(std::ostream& OS, char* buf, int k, int r, int j, int h) {
  sprintf(buf, "%02d%02d%s%s%06d", 0, 0, FillZero(ToBinary(k*r+j), 14).c_str(), FillZero(ToBinary(h), 5).c_str(), 0);
  OS << "0x" << std::setfill('0') << std::setw(8) << std::hex << std::stoi(buf, nullptr, 2) << " COMP\n";
  OS.flush();
}

void ReadRes(std::ostream& OS) {
  OS << "READRES\n";
  OS.flush();
}

void OutputNewtonTrace(std::ostream& OS, std::string kernel_name, int64_t row, int64_t col) {
  char buf[100]; // fixed size

  int num_chunks = (col + 511) / 512;
  int r = (row + 15) / 16;

  for (int i = 0; i < num_chunks; i++) {
    GWrite(OS, buf, i);
    for (int j = 0; j < r; j++) {
      int num_act = 4;
      if (j == r - 1 && row % 16 != 0) {
        num_act = ((row - (row/16)*16) + 3) / 4;
      }
      GAct(OS, buf, i, r, j, num_act);
      int bound = 32;
      if (i == num_chunks - 1 && col % 512 != 0) {
        bound = ((col - (col/512)*512) + 15) / 16;
      }
      for (int k = 0; k < bound; k++) {
        Comp(OS, buf, i, r, j, k);
      }
      ReadRes(OS);
    }
  }
  OS.flush();
}

void OutputNewtonTraceV2(std::ostream& OS, std::string kernel_name, int64_t row, int64_t col, int64_t stride, StrideInfo sinfo) {
  char buf[100]; // fixed size

  int num_chunks = (col + 511) / 512;
  int r = (row + 15) / 16;

  for (int i = 0; i < num_chunks; i++) {
    int elem = 0;
    for (int g = 0; g < sinfo.num_gwrite; g++) {
      GWrite(OS, buf, i, sinfo);
      break; // TODO: allow multiple gwrites
    }
    for (int j = 0; j < r; j++) {
      int num_act = 4;
      // if (j == r - 1 && row % 16 != 0) {
      //   num_act = ((row - (row/16)*16) + 3) / 4;
      // }
      GAct(OS, buf, i, r, j, num_act);
      int bound = 32;
      if (col % 512 != 0) {
        bound = ((col - (col/512)*512) + 15) / 16;
      }
      for (int k = 0; k < bound; k++) {
        Comp(OS, buf, i, r, j, k);
        elem += 16;

        if (k == bound - 1 || elem % stride == 0) {
          ReadRes(OS);
        }
      }
    }
  }
  OS.flush();
}
}
