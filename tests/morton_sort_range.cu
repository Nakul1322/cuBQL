// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/builder/cuda/radix.h"
#include "data/eda_layout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t result)
{
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}

template<typename T>
struct DeviceBuffer {
  T *data = nullptr;
  explicit DeviceBuffer(size_t count) {
    check(cudaMalloc(reinterpret_cast<void **>(&data),count*sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

cudaError_t freeTreeBuffer(void *ptr)
{
#if CUDART_VERSION >= 11020
  return cudaFreeAsync(ptr,0);
#else
  return cudaFree(ptr);
#endif
}

template<typename T, int D>
struct BuiltTree {
  cuBQL::BinaryBVH<T,D> bvh;
  BuiltTree() = default;
  BuiltTree(const BuiltTree &) = delete;
  BuiltTree &operator=(const BuiltTree &) = delete;
  ~BuiltTree() noexcept {
    if (bvh.nodes) freeTreeBuffer(bvh.nodes);
    if (bvh.primIDs) freeTreeBuffer(bvh.primIDs);
  }
  void release() {
    if (bvh.nodes) { check(freeTreeBuffer(bvh.nodes)); bvh.nodes = nullptr; }
    if (bvh.primIDs) { check(freeTreeBuffer(bvh.primIDs)); bvh.primIDs = nullptr; }
  }
};

struct Event {
  cudaEvent_t handle;
  Event() { check(cudaEventCreate(&handle)); }
  ~Event() noexcept { cudaEventDestroy(handle); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
};

template<typename T, int D>
struct HostTree {
  std::vector<typename cuBQL::BinaryBVH<T,D>::node_t> nodes;
  std::vector<uint32_t> primIDs;
};

template<typename T, int D>
HostTree<T,D> snapshot(const cuBQL::BinaryBVH<T,D> &bvh)
{
  HostTree<T,D> result;
  result.nodes.resize(bvh.numNodes);
  result.primIDs.resize(bvh.numPrims);
  check(cudaMemcpy(result.nodes.data(),bvh.nodes,
                   result.nodes.size()*sizeof(result.nodes[0]),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(result.primIDs.data(),bvh.primIDs,
                   result.primIDs.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
  return result;
}

__global__ void auditKeyBits(const uint64_t *keys, uint32_t count,
                             int bits, unsigned *bad)
{
  const uint32_t i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < count && (keys[i] >> bits) != 0) atomicExch(bad,1u);
}

template<typename T, int D>
void auditGeneratedKeys(const cuBQL::box_t<T,D> *boxes, uint32_t count)
{
  namespace radix = cuBQL::radixBuilder_impl;
  constexpr int bits = D*radix::numMortonBits<D>::value;
  static_assert(bits > 0 && bits < 64);
  DeviceBuffer<radix::BuildState<T,D>> state(1);
  DeviceBuffer<uint64_t> keys(count);
  DeviceBuffer<uint32_t> ids(count);
  DeviceBuffer<unsigned> bad(1);
  check(cudaMemset(bad.data,0,sizeof(unsigned)));
  radix::clearBuildState<T,D><<<32,1>>>(state.data,count);
  radix::fillBuildState<T,D><<<(count+1023)/1024,1024>>>(state.data,boxes,count);
  radix::finishBuildState<T,D><<<32,1>>>(state.data);
  radix::computeUnsortedKeysAndPrimIDs<T,D><<<(count+1023)/1024,1024>>>
    (keys.data,ids.data,state.data,boxes,count);
  auditKeyBits<<<(count+255)/256,256>>>(keys.data,count,bits,bad.data);
  check(cudaGetLastError());
  unsigned badValue = 0;
  check(cudaMemcpy(&badValue,bad.data,sizeof(badValue),cudaMemcpyDeviceToHost));
  if (badValue) throw std::runtime_error("Morton key exceeds meaningful sort range");
}

template<typename T, int D>
void compareTrees(const HostTree<T,D> &a, const HostTree<T,D> &b, size_t count)
{
  if (a.nodes.size() != b.nodes.size() || a.primIDs != b.primIDs ||
      a.primIDs.size() != count)
    throw std::runtime_error("BVH size or sorted primitive IDs differ");
  std::vector<unsigned char> seen(count,0), covered(count,0);
  for (uint32_t id : a.primIDs) {
    if (id >= count || seen[id]++)
      throw std::runtime_error("primitive IDs are not a permutation of the input");
  }
  std::vector<std::pair<uint32_t,uint32_t>> pending{{0,0}};
  size_t visited = 0;
  while (!pending.empty()) {
    const auto pair = pending.back(); pending.pop_back();
    if (pair.first >= a.nodes.size() || pair.second >= b.nodes.size() ||
        ++visited > a.nodes.size())
      throw std::runtime_error("invalid BVH topology");
    const auto &left = a.nodes[pair.first];
    const auto &right = b.nodes[pair.second];
    if (left.admin.count != right.admin.count)
      throw std::runtime_error("leaf counts differ");
    for (int axis=0;axis<D;++axis) {
      const T lowerA = left.bounds.lower[axis], lowerB = right.bounds.lower[axis];
      const T upperA = left.bounds.upper[axis], upperB = right.bounds.upper[axis];
      if (std::memcmp(&lowerA,&lowerB,sizeof(T)) || std::memcmp(&upperA,&upperB,sizeof(T)))
        throw std::runtime_error("node bounds differ");
    }
    if (left.admin.count) {
      const uint64_t end = left.admin.offset+left.admin.count;
      if (left.admin.offset != right.admin.offset || end > count)
        throw std::runtime_error("leaf ranges differ or exceed primitive array");
      for (uint64_t i=left.admin.offset;i<end;++i)
        if (covered[size_t(i)]++) throw std::runtime_error("overlapping leaf ranges");
    } else {
      pending.emplace_back(uint32_t(left.admin.offset)+1,uint32_t(right.admin.offset)+1);
      pending.emplace_back(uint32_t(left.admin.offset),uint32_t(right.admin.offset));
    }
  }
  if (visited+1 != a.nodes.size() || std::find(covered.begin(),covered.end(),0) != covered.end())
    throw std::runtime_error("unreachable BVH nodes or primitives");
}

struct QueryResult { double area; uint32_t hits; unsigned overflow; };

__device__ bool positiveOverlap(const cuBQL::box_t<float,2> &a,
                                const cuBQL::box_t<float,2> &b)
{
  return a.lower[0] < b.upper[0] && b.lower[0] < a.upper[0] &&
         a.lower[1] < b.upper[1] && b.lower[1] < a.upper[1];
}

__global__ void queryAreas(cuBQL::BinaryBVH<float,2> bvh,
                          const cuBQL::box_t<float,2> *boxes,
                          const cuBQL::box_t<float,2> *queries,
                          QueryResult *results, uint32_t count)
{
  const uint32_t q = blockIdx.x*blockDim.x+threadIdx.x;
  if (q >= count) return;
  uint32_t stack[64], depth = 1;
  stack[0] = 0;
  QueryResult result{0,0,0};
  while (depth) {
    const auto node = bvh.nodes[stack[--depth]];
    if (!positiveOverlap(node.bounds,queries[q])) continue;
    if (!node.admin.count) {
      if (depth+2 > 64) { result.overflow = 1; break; }
      stack[depth++] = uint32_t(node.admin.offset)+1;
      stack[depth++] = uint32_t(node.admin.offset);
    } else {
      for (uint32_t i=0;i<node.admin.count;++i) {
        const auto box = boxes[bvh.primIDs[node.admin.offset+i]];
        if (!positiveOverlap(box,queries[q])) continue;
        const float dx = fminf(box.upper[0],queries[q].upper[0])-fmaxf(box.lower[0],queries[q].lower[0]);
        const float dy = fminf(box.upper[1],queries[q].upper[1])-fmaxf(box.lower[1],queries[q].lower[1]);
        result.area += double(dx)*double(dy);
        ++result.hits;
      }
    }
  }
  results[q] = result;
}

void auditQueries(const std::vector<cuBQL::box_t<float,2>> &boxes,
                  const cuBQL::box_t<float,2> *deviceBoxes,
                  const cuBQL::BinaryBVH<float,2> &a,
                  const cuBQL::BinaryBVH<float,2> &b)
{
  using Box = cuBQL::box_t<float,2>;
  const eda_layout::Rectangle windows[] = {
    {-140,-140,-100,-90}, {-128,-128,0,0}, {-120,-116,-116,-92},
    {-16,-16,16,16}, {0,0,64,64}, {48,54,56,62}, {112,112,124,124},
    {-256,-256,-200,-200}, {-128,-128,896,384}, {100,4,104,40}
  };
  std::vector<Box> queries;
  for (const auto &r : windows) {
    Box box;
    box.lower[0] = float(r.x0); box.upper[0] = float(r.x1);
    box.lower[1] = float(r.y0); box.upper[1] = float(r.y1);
    queries.push_back(box);
  }
  DeviceBuffer<Box> deviceQueries(queries.size());
  DeviceBuffer<QueryResult> deviceResults(queries.size());
  check(cudaMemcpy(deviceQueries.data,queries.data(),queries.size()*sizeof(Box),cudaMemcpyHostToDevice));
  for (const auto &tree : {a,b}) {
    queryAreas<<<1,32>>>(tree,deviceBoxes,deviceQueries.data,deviceResults.data,uint32_t(queries.size()));
    check(cudaGetLastError());
    std::vector<QueryResult> results(queries.size());
    check(cudaMemcpy(results.data(),deviceResults.data,results.size()*sizeof(QueryResult),cudaMemcpyDeviceToHost));
    for (size_t q=0;q<queries.size();++q) {
      int64_t area = 0;
      uint32_t hits = 0;
      for (const auto &box : boxes) {
        const int64_t dx = std::min(int64_t(box.upper[0]),int64_t(queries[q].upper[0]))-
                           std::max(int64_t(box.lower[0]),int64_t(queries[q].lower[0]));
        const int64_t dy = std::min(int64_t(box.upper[1]),int64_t(queries[q].upper[1]))-
                           std::max(int64_t(box.lower[1]),int64_t(queries[q].lower[1]));
        if (dx > 0 && dy > 0) { area += dx*dy; ++hits; }
      }
      if (results[q].overflow || results[q].area != double(area) || results[q].hits != hits)
        throw std::runtime_error("2D query differs from independent CPU area/overlap oracle");
    }
  }
  std::cout << "query_oracle=PASS queries=" << queries.size() << "\n";
}

template<typename T, int D>
void auditBuilds(const std::vector<cuBQL::box_t<T,D>> &boxes,
                 const cuBQL::box_t<T,D> *deviceBoxes)
{
  auditGeneratedKeys<T,D>(deviceBoxes,uint32_t(boxes.size()));
  BuiltTree<T,D> full, narrowed;
  cuBQL::BuildConfig config(16);
  cuBQL::cuda::radixBuilder(full.bvh,deviceBoxes,uint32_t(boxes.size()),config);
  config.enableMeaningfulMortonBitSort();
  cuBQL::cuda::radixBuilder(narrowed.bvh,deviceBoxes,uint32_t(boxes.size()),config);
  check(cudaDeviceSynchronize());
  compareTrees(snapshot(full.bvh),snapshot(narrowed.bvh),boxes.size());
  if constexpr (std::is_same<T,float>::value && D == 2)
    auditQueries(boxes,deviceBoxes,full.bvh,narrowed.bvh);
  full.release();
  narrowed.release();
}

const char *patternName(eda_layout::Pattern pattern)
{
  return pattern == eda_layout::Pattern::Layout ? "layout" : "motif";
}

template<typename T, int D>
void runCase(const char *name, eda_layout::Pattern pattern)
{
  const auto boxes = eda_layout::makeBoxes<T,D>(8,4,pattern);
  DeviceBuffer<cuBQL::box_t<T,D>> deviceBoxes(boxes.size());
  check(cudaMemcpy(deviceBoxes.data,boxes.data(),boxes.size()*sizeof(boxes[0]),cudaMemcpyHostToDevice));
  auditBuilds(boxes,deviceBoxes.data);
  std::cout << "geometry=" << patternName(pattern) << " case=" << name
            << " boxes=" << boxes.size() << " geometry_fnv64="
            << std::hex << eda_layout::fingerprint(boxes) << std::dec << " status=PASS\n";
}

void runCases(eda_layout::Pattern pattern)
{
  runCase<float,2>("float2",pattern); runCase<float,3>("float3",pattern); runCase<float,4>("float4",pattern);
  runCase<double,2>("double2",pattern); runCase<double,3>("double3",pattern); runCase<double,4>("double4",pattern);
  runCase<int,2>("int2",pattern); runCase<int,3>("int3",pattern); runCase<int,4>("int4",pattern);
  runCase<int64_t,2>("int64_2",pattern); runCase<int64_t,3>("int64_3",pattern); runCase<int64_t,4>("int64_4",pattern);
}

float timedBuild(const cuBQL::box_t<float,2> *boxes, uint32_t count, bool narrow)
{
  BuiltTree<float,2> tree;
  cuBQL::BuildConfig config(16);
  config.enableMeaningfulMortonBitSort(narrow);
  Event start, stop;
  check(cudaEventRecord(start.handle));
  cuBQL::cuda::radixBuilder(tree.bvh,boxes,count,config);
  check(cudaGetLastError());
  check(cudaEventRecord(stop.handle)); check(cudaEventSynchronize(stop.handle));
  float ms = 0;
  check(cudaEventElapsedTime(&ms,start.handle,stop.handle));
  if (!std::isfinite(ms) || ms <= 0)
    throw std::runtime_error("CUDA event interval is not finite and positive");
  tree.release();
  return ms;
}

void benchmark(uint32_t tilesX, uint32_t tilesY, eda_layout::Pattern pattern)
{
  constexpr int meaningfulBits = 2*cuBQL::radixBuilder_impl::numMortonBits<2>::value;
  const auto boxes = eda_layout::makeBoxes<float,2>(tilesX,tilesY,pattern);
  cudaDeviceProp device{};
  int currentDevice = 0;
  check(cudaGetDevice(&currentDevice)); check(cudaGetDeviceProperties(&device,currentDevice));
  int runtimeVersion = 0, driverVersion = 0;
  check(cudaRuntimeGetVersion(&runtimeVersion)); check(cudaDriverGetVersion(&driverVersion));
  std::cout << "geometry=" << patternName(pattern) << " benchmark_device=" << device.name << "\n"
            << "compute_capability=" << device.major << '.' << device.minor
            << " device_bytes=" << device.totalGlobalMem
            << " cuda_runtime=" << runtimeVersion << " cuda_driver=" << driverVersion << "\n"
            << "tiles=" << tilesX << 'x' << tilesY << " boxes=" << boxes.size()
            << " input_bytes=" << boxes.size()*sizeof(boxes[0]) << " geometry_fnv64="
            << std::hex << eda_layout::fingerprint(boxes) << std::dec
            << " leaf=16 warmups_per_arm=1 balanced_pairs=6\n";
  DeviceBuffer<cuBQL::box_t<float,2>> deviceBoxes(boxes.size());
  check(cudaMemcpy(deviceBoxes.data,boxes.data(),boxes.size()*sizeof(boxes[0]),cudaMemcpyHostToDevice));
  auditBuilds(boxes,deviceBoxes.data);
  timedBuild(deviceBoxes.data,uint32_t(boxes.size()),false);
  timedBuild(deviceBoxes.data,uint32_t(boxes.size()),true);
  std::vector<float> times[2];
  for (int round=0;round<6;++round)
    for (int position=0;position<2;++position) {
      const int arm = (round+position)%2;
      const float ms = timedBuild(deviceBoxes.data,uint32_t(boxes.size()),arm != 0);
      times[arm].push_back(ms);
      std::cout << "round=" << round+1 << " position=" << position+1
                << " end_bit=" << (arm ? meaningfulBits : 64) << " build_ms=" << ms << "\n";
    }
  for (auto &arm : times) std::sort(arm.begin(),arm.end());
  const float full = (times[0][2]+times[0][3])*0.5f;
  const float narrow = (times[1][2]+times[1][3])*0.5f;
  std::cout << "geometry=" << patternName(pattern)
            << " median_full64_ms=" << full << " median_meaningful_ms=" << narrow
            << " speedup=" << full/narrow << " validation=PASS\n";
}

uint32_t parseTileCount(const char *text, const char *name)
{
  const std::string error = std::string(name)+" must be a positive integer";
  uint32_t count = 0;
  if (!*text) throw std::invalid_argument(error);
  for (const char *p=text;*p;++p) {
    if (*p < '0' || *p > '9') throw std::invalid_argument(error);
    const uint32_t digit = uint32_t(*p-'0');
    if (count > (std::numeric_limits<uint32_t>::max()-digit)/10)
      throw std::invalid_argument(error);
    count = count*10+digit;
  }
  if (!count) throw std::invalid_argument(error);
  return count;
}

void validateBenchmarkSize(uint32_t tilesX, uint32_t tilesY)
{
  constexpr uint32_t maxTiles = (1u<<24)/eda_layout::tilePitch;
  if (tilesX > maxTiles || tilesY > maxTiles)
    throw std::invalid_argument("tile dimensions exceed exact float coordinate range");
  const uint64_t count = uint64_t(tilesX)*tilesY*eda_layout::rectanglesPerTile;
  if (count > uint64_t(std::numeric_limits<int>::max()/2))
    throw std::invalid_argument("box count exceeds radix builder integer range");
  if (count > std::numeric_limits<size_t>::max()/sizeof(cuBQL::box_t<float,2>))
    throw std::invalid_argument("input exceeds host address range");
}

void printHelp(const char *program)
{
  std::cout << "Usage: " << program << " [--help | --benchmark [tilesX tilesY]\n"
            << "       | --benchmark-motif [tilesX tilesY] | --write-layout output.svg]\n"
            << "  No arguments: check 64-bit versus meaningful-bit sorting for 12 scalar/dimension\n"
            << "    cases on each geometry pattern (24 cases, 1024 boxes each).\n"
            << "  --benchmark: generate an original layout-inspired pattern; default 256x256 tiles\n"
            << "    produce 2097152 float2 boxes (32 MiB input). Use --benchmark 8 4 for a small run.\n"
            << "  --benchmark-motif: regenerate the historical A100 motif; default 1024x512 tiles\n"
            << "    produce 16777216 boxes (256 MiB input). Use --benchmark-motif 8 4 for a small run.\n"
            << "    Both benchmarks validate trees and queries, warm both sort modes, then time six\n"
            << "    balanced pairs of complete radix-builder calls with CUDA events (leaf threshold 16).\n"
            << "    Both sort modes use uint64_t keys and the same quantization; only the CUB end bit\n"
            << "    changes (64 versus currently 30 in 2D). No speedup threshold is required.\n"
            << "  --write-layout: write a 16x8 preview of the new layout pattern without initializing\n"
            << "    CUDA. Both geometries use deterministic integer coordinates generated in C++;\n"
            << "    no image input or external layout data is needed.\n";
}

} // namespace

int main(int argc, char **argv)
{
  try {
    std::cout << std::fixed << std::setprecision(6);
    if (argc == 2 && std::strcmp(argv[1],"--help") == 0) {
      printHelp(argv[0]);
    } else if (argc == 3 && std::strcmp(argv[1],"--write-layout") == 0) {
      eda_layout::writeSvg(argv[2],16,8);
      std::cout << "geometry=layout layout_svg=" << argv[2] << " tiles=16x8 boxes="
                << 16*8*eda_layout::rectanglesPerTile << "\n";
    } else if ((argc == 2 || argc == 4) &&
               (std::strcmp(argv[1],"--benchmark") == 0 || std::strcmp(argv[1],"--benchmark-motif") == 0)) {
      const bool motif = std::strcmp(argv[1],"--benchmark-motif") == 0;
      const uint32_t tilesX = argc == 4 ? parseTileCount(argv[2],"tilesX") : (motif ? 1024 : 256);
      const uint32_t tilesY = argc == 4 ? parseTileCount(argv[3],"tilesY") : (motif ? 512 : 256);
      validateBenchmarkSize(tilesX,tilesY);
      benchmark(tilesX,tilesY,motif ? eda_layout::Pattern::Motif : eda_layout::Pattern::Layout);
    } else if (argc == 1) {
      runCases(eda_layout::Pattern::Layout);
      runCases(eda_layout::Pattern::Motif);
    } else {
      throw std::invalid_argument("invalid arguments; use --help for usage");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Morton sort-range test failed: " << error.what() << '\n';
    return 1;
  }
}
