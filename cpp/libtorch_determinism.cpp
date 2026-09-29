#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <cstdint>

extern "C" void dope_enable_libtorch_determinism() {
  auto &context = at::globalContext();
  context.setDeterministicAlgorithms(true, false);
  context.setDeterministicCuDNN(true);
  context.setDeterministicFillUninitializedMemory(true);
}

extern "C" void dope_reset_cuda_peak_memory() {
  c10::cuda::CUDACachingAllocator::resetPeakStats(0);
}

extern "C" std::uint64_t dope_cuda_peak_memory_bytes() {
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  constexpr auto aggregate =
      static_cast<std::size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  return static_cast<std::uint64_t>(
      std::max(stats.allocated_bytes[aggregate].peak,
               stats.reserved_bytes[aggregate].peak));
}
