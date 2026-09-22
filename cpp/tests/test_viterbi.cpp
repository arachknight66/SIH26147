#include "sih/core/types.hpp"
#include "sih/fec/viterbi.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> llrs_from_bits(const std::span<const std::uint8_t> bits) {
    std::vector<float> result;
    result.reserve(bits.size());
    for (const auto bit : bits) {
        result.push_back(bit == 1U ? 8.0F : -8.0F);
    }
    return result;
}

}  // namespace

int main() {
    // Fixed K=7 (171,133 octal), zero-state vector checked independently by hand.
    constexpr std::array<std::uint8_t, 8> expected{1, 0, 1, 1, 0, 0, 1, 1};
    constexpr std::array<std::uint8_t, 16> encoded{
        1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0};
    const auto llrs = llrs_from_bits(encoded);
    const auto progress = std::make_shared<sih::core::ProgressState>();
    const auto result = sih::fec::decode_k7_r12_soft(llrs, {}, progress);

    assert(result.execution_status == sih::core::ExecutionStatus::completed);
    assert(result.decode_validity == sih::core::EvidenceStatus::unverified);
    assert(result.consumed_llrs == encoded.size());
    assert(result.residual_llrs == 0);
    assert(result.decoded_bits->size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        assert((*result.decoded_bits)[index] == expected[index]);
    }
    assert(progress->fraction() == 1.0);

    bool odd_rejected = false;
    try {
        const std::array<float, 1> odd{1.0F};
        static_cast<void>(sih::fec::decode_k7_r12_soft(odd));
    } catch (const std::invalid_argument&) {
        odd_rejected = true;
    }
    assert(odd_rejected);

    bool nonfinite_rejected = false;
    try {
        const std::array<float, 2> nonfinite{
            1.0F, std::numeric_limits<float>::quiet_NaN()};
        static_cast<void>(sih::fec::decode_k7_r12_soft(nonfinite));
    } catch (const std::invalid_argument&) {
        nonfinite_rejected = true;
    }
    assert(nonfinite_rejected);

    const auto cancellation = std::make_shared<sih::core::CancellationToken>();
    cancellation->cancel();
    const auto cancelled = sih::fec::decode_k7_r12_soft(llrs, cancellation);
    assert(cancelled.execution_status == sih::core::ExecutionStatus::cancelled);
    assert(cancelled.consumed_llrs == 0);
    assert(cancelled.residual_llrs == llrs.size());
    assert(cancelled.decoded_bits->empty());
}
