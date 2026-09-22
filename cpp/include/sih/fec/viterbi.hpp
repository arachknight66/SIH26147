#pragma once

#include "sih/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace sih::fec {

struct ViterbiResult {
    std::shared_ptr<std::vector<std::uint8_t>> decoded_bits{
        std::make_shared<std::vector<std::uint8_t>>()};
    core::ExecutionStatus execution_status{core::ExecutionStatus::completed};
    core::EvidenceStatus decode_validity{core::EvidenceStatus::unverified};
    float path_metric_margin{0.0F};
    std::size_t consumed_llrs{0};
    std::size_t residual_llrs{0};
};

[[nodiscard]] ViterbiResult decode_k7_r12_soft(
    std::span<const float> llrs,
    const std::shared_ptr<core::CancellationToken>& cancellation = {},
    const std::shared_ptr<core::ProgressState>& progress = {});

}  // namespace sih::fec
