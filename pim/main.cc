#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <string>
#include <cassert>
#include <iostream>

#include "pim_trace.h"

using Str2StrMap = std::unordered_map<std::string, std::string>;

int nextPow2(int n) {
  int power = 1;
  while (power < n) {
    power *= 2;
  }
  return power;
}

std::vector<std::string> PimCodeGen(std::string id, const Str2StrMap& attrs,
                const std::vector<std::string>& func_args) {
  int n_channel = std::stoi(attrs.at("N_CHANNEL"));
  std::vector<std::string> code;
  std::ostringstream OS;

  int h = std::stoi(attrs.at("H"));
  int w = std::stoi(attrs.at("W"));
  int kh = std::stoi(attrs.at("KH"));
  int kw = std::stoi(attrs.at("KW"));
  int ph = std::stoi(attrs.at("PH"));
  int pw = std::stoi(attrs.at("PW"));
  int stride_ = std::stoi(attrs.at("S"));

  int H = (h - kh + 2 * ph) / stride_ + 1;
  int W = (w - kw + 2 * pw) / stride_ + 1;
  int C_o = std::stoi(attrs.at("K"));
  int C_i = std::stoi(attrs.at("C"));

  auto i_c = (((C_i * kh * kw) + 15) / 16) * 16;
  auto o_c = C_o;
  auto row = ((o_c + 15) / 16) * 16;
  int fl, stride, n, col;

  // NOTE: Boundary condition is ignored (e.g., 3x3 kernel with (1, 1) padding)
  if (i_c <= 512) {
    fl = 512 / i_c;
    col = i_c * fl;
    stride = i_c;
    n = ((H * W) + fl - 1) / fl;
    for (int i = 0; i < n; i++) {
      pim::StrideInfo sinfo;
      if (kh > 1 || kw > 1) {
        sinfo.use_stride = true;
        sinfo.num_first_elem = C_i * kh;
        sinfo.stride = C_i * (h - kh);
        sinfo.num_after_elem = C_i * kh;
        sinfo.num_gwrite = 512 / (C_i * kh * kw);
      }
      pim::OutputNewtonTraceV2(OS, id, row, col, stride, sinfo);
    }
    code.push_back(OS.str());
  } else {
    int full = i_c / 512;
    col = 512;
    stride = 512;
    n = (H * W);
    // n = std::min(((n + n_channel - 1) / n_channel) * n_channel, nextPow2(n));
    for (int j = 0; j < full; j++) {
      for (int i = 0; i < n; i++) {
        pim::OutputNewtonTraceV2(OS, id, row, col, stride);
      }
    }
    code.push_back(OS.str());
    OS.str("");
    OS.clear();
    int i_c_remain = i_c - (512 * full);
    if (i_c_remain > 0) {
      // i_c_remain = nextPow2(i_c_remain);
      i_c_remain = ((i_c_remain + 15) / 16) * 16;
      fl = 512 / i_c_remain;
      col = fl * i_c_remain;
      stride = i_c_remain;
      n = (((H * W) + fl - 1) / fl);
      // n = std::min(((n + n_channel - 1) / n_channel) * n_channel, nextPow2(n));
      for (int i = 0; i < n; i++) {
        pim::OutputNewtonTraceV2(OS, id, row, col, stride);
      }
    }
    code.push_back(OS.str());
  }
  return code;
}

class Readres {
  std::vector<std::string> head;

public:
  Readres(std::vector<std::string> cmds) {
    head = cmds;
  }

  std::string code() const {
    std::string code;
    for (auto h : head) {
      code += h + "\n";
    }
    return code;
  }

  int n_comp() {
    if (head.size() == 0)
      return 0;
    return head.size() - 1;
  }

  std::string at(int i) {
    return head[i];
  }
};

class GAct {
  std::vector<std::string> head;
  std::vector<Readres> readres;
  std::vector<std::string> buffer;

public:
  GAct(std::string cmd) {
    head.push_back(cmd);
  }
  void add(std::string cmd) {
    if (cmd.find("G_ACT") != std::string::npos) {
      head.push_back(cmd);
    } else {
      buffer.push_back(cmd);
      if (cmd.find("READRES") != std::string::npos) {
        readres.push_back(Readres(buffer));
        buffer.clear();
      }
    }
  }

  Readres& at(int i) {
    return readres[i];
  }

  int n_readres() const {
    return readres.size();
  }

  std::string h() const {
    std::string code;
    for (auto& h : head) {
      code += h + "\n";
    }
    return code;
  }

  std::string code(bool include_head) const {
    std::string code;
    if (include_head) {
      for (auto h : head) {
        code += h + "\n";
      }
    }
    for (auto& r : readres) {
      code += r.code();
    }
    return code;
  }
  void split(int factor) {
    std::vector<Readres> new_readres;
    for (int i = 0, offset = 0; i < n_readres(); i++, offset++) {
      Readres& rr = readres[i];
      int chunk = rr.n_comp() / factor;
      for (int j = 0; j < factor; j++) {
        std::vector<std::string> buf;
        for (int k = 0; k < chunk; k++) {
          if (j*chunk + k + offset >= rr.n_comp()) {
            // std::cerr << "WARN: imbalance scheduling!" << std::endl;
            continue;
          }
          buf.push_back(rr.at(j*chunk + k + offset));
        }
        buf.push_back("READRES");
        new_readres.push_back(Readres(buf));
      }
    }
    readres = new_readres;
  }
};

class GWrite {
  std::string head;
  std::vector<GAct> gacts;

public:
  GWrite(std::string cmd) : head(cmd) { }

  void add(std::string cmd) {
    if (cmd.find("G_ACT0") != std::string::npos) {
      gacts.push_back(GAct(cmd));
    } else {
      auto& gact = gacts.back();
      gact.add(cmd);
    }
  }

  int n_gact() const {
    return gacts.size();
  }

  int n_readres() const {
    int n = 0;
    for (auto& gact : gacts) {
      n += gact.n_readres();
    }
    return n;
  }

  GAct& at(int i) {
    return gacts[i];
  }

  std::string h() const {
    return head + "\n";
  }

  std::string code(bool include_head) const {
    std::string code;
    if (include_head) {
      code += head + "\n";
    }
    for (auto& gact : gacts) {
      code += gact.code(true);
    }
    return code;
  }
};

class Command {
  std::vector<GWrite> gwrites;
  int n_channel;

public:
  Command(int n_channel) : n_channel(n_channel) { }

  void add(std::string cmd) {
    if (cmd.find("GWRITE") != std::string::npos) {
      gwrites.push_back(GWrite(cmd));
    } else {
      auto& gwrite = gwrites.back();
      gwrite.add(cmd);
    }
  }

  void policy_readres_auto(GWrite& gwrite, std::vector<std::string>& code, int n, int offset=0, int n_gwrite=1) {
    // write GWRITE for every channels
    for (int i = 0; i < n; i++) {
      std::string g = gwrite.h();
      int s = g.find("GWRITE_");
      if (s == std::string::npos) {
        // replace regular GWRITE with multiple version (GWRITE_2/GWRITE_4)
        g = g.substr(0, g.find("GWRITE")) + std::string("GWRITE_") + std::to_string(n_gwrite) + g.substr(g.find("GWRITE") + 6, g.length());
      }
      code[i + offset] += g;
    }

    // TODO: pick right gact for validation by value
    GAct& gact = gwrite.at(0);

    // distribute readres
    int parallelism = gwrite.n_gact() * gact.n_readres();

    // exploit finer-grained parallelism at the expense of energy increase.
    while (parallelism <= n / 2) {
      int factor = n / parallelism;
      gact.split(factor);
      parallelism = gwrite.n_gact() * gact.n_readres();
    }

    // # READRES per G_ACT
    int stride = std::min(
      std::max((gwrite.n_gact() * gact.n_readres() + n - 1) / n, 1),
      gact.n_readres());
    int gw_n_readres = gwrite.n_gact() * gact.n_readres();
    for (int j = 0, idx = 0; j < gw_n_readres; j += stride, idx++) {
      code[idx % n + offset] += gact.h();
      for (int k = 0; k < stride; k++) {
        if (j + k >= gw_n_readres) {
          break;
        }
        auto& rr = gact.at(gact.n_readres() - 1);
        if (j + k < gw_n_readres) {
          // TODO: pick right rr for validation by value, considering remainder
          // e.g., ./pim_codegen -oc 32 -ic 3 -h 113 -w 224 -kh 3 -kw 3 -ph 0 -pw 1 -stride 2 -name test12 -gw 4 -n_channel 12
          rr = gact.at(k);
        }
        code[idx % n + offset] += rr.code();
      }
    }
  }
  std::vector<std::string> policy_auto(const Str2StrMap& attrs) {
    std::vector<std::string> best_code(n_channel);
    for (int n_gwrite = 1; n_gwrite <= std::stoi(attrs.at("GW")); n_gwrite *= 2) {
      std::vector<std::string> code(n_channel);
      int stride = n_channel / n_gwrite;
      int chunk = (gwrites.size() + n_gwrite - 1) / n_gwrite;
      for (int i = 0; i < n_gwrite; i++) {
        int offset = i * stride;
        for (int j = 0; j < chunk; j++) {
          if (i * chunk + j >= gwrites.size()) {
            break;
          }
          auto& gwrite = gwrites[i * chunk + j];
          // TODO: for valication by value, (stride) number of gwrites must be passed to the policy_readres_auto
          policy_readres_auto(gwrite, code, stride, offset, n_gwrite);
        }
      }
      std::string::size_type pos = 0;
      int gact_best = 0;
      while (true) {
        pos = best_code[0].find("G_ACT0", pos);
        if (pos == std::string::npos) {
          pos = 0;
          break;
        }
        ++gact_best;
        ++pos;
      }
      int gact_code = 0;
      while (true) {
        pos = code[0].find("G_ACT0", pos);
        if (pos == std::string::npos) {
          pos = 0;
          break;
        }
        ++gact_code;
        ++pos;
      }
      if (best_code[0].size() == 0 || gact_code < gact_best || gact_code == gact_best && code[0].size() < best_code[0].size()) {
        best_code = code;
      }
    }
    return best_code;
  }
};

void PimSchedule(std::string id, const Str2StrMap& attrs,
                 const std::vector<std::string>& func_args, std::string code, bool append=false) {
  int n_channel = std::stoi(attrs.at("N_CHANNEL"));
  int gpu_channel = 32 - n_channel;
  auto mode = std::ios_base::out;
  if (append) {
    mode = std::ios_base::app;
  }

  std::vector<std::string> traces;
  std::string token;
  std::stringstream ss(code);

  Command command(n_channel);

  int idx = 0;
  while (std::getline(ss, token, '\n')) {
    traces.push_back(token);
    command.add(token);
    idx++;
  }

  std::ofstream OS;

  OS.open(id + "-all.pim", mode);
  for (auto trace : traces) {
    OS << trace << "\n";
  }
  OS.flush();
  OS.close();

  std::vector<std::string> cmds = command.policy_auto(attrs);

  for (int i = gpu_channel; i < gpu_channel + n_channel; i++) {
    OS.open(id + "-" + std::to_string(i) + ".pim", mode);
    OS << cmds[i - gpu_channel];
    OS.flush();
    OS.close();
  }
}

bool isdigit(const char *str) {
  for (int i = 0; i < strlen(str); ++i)
    if (str[i] < '0' || str[i] > '9')
      return false;
  return true;
}

void get_args(int argc, char *argv[], Str2StrMap& attrs) {
  int pos = 1;
  // set default value
  attrs["GW"] = "4";
  attrs["N_CHANNEL"] = "16";

  while (pos < argc) {
    if (pos + 1 < argc && !strcmp(argv[pos], "-oc") &&
      isdigit(argv[pos + 1])) {
      attrs["K"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-ic") &&
      isdigit(argv[pos + 1])) {
      attrs["C"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-h") &&
      isdigit(argv[pos + 1])) {
      attrs["H"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-w") &&
      isdigit(argv[pos + 1])) {
      attrs["W"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-kh") &&
      isdigit(argv[pos + 1])) {
      attrs["KH"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-kw") &&
      isdigit(argv[pos + 1])) {
      attrs["KW"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-ph") &&
      isdigit(argv[pos + 1])) {
      attrs["PH"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-pw") &&
      isdigit(argv[pos + 1])) {
      attrs["PW"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-stride") &&
      isdigit(argv[pos + 1])) {
      attrs["S"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-name")) {
      attrs["name"] = argv[pos + 1];
      pos += 2;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-gw") &&
      isdigit(argv[pos + 1])) {
      attrs["GW"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (pos + 1 < argc && !strcmp(argv[pos], "-n_channel") &&
      isdigit(argv[pos + 1])) {
      attrs["N_CHANNEL"] = argv[pos + 1];
      pos += 2;
      continue;
    }
    pos += 1;
  }
}

int main(int argc, char *argv[]) {
  Str2StrMap attrs{};
  get_args(argc, argv, attrs);
  std::string kernel_name = attrs.at("name");
  auto code = PimCodeGen(kernel_name, attrs, {});

  for (int i = 0; i < code.size(); i++) {
    PimSchedule(kernel_name, attrs, {}, code[i], i > 0);
  }
  return 0;
}
