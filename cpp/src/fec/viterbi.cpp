#include "sih/fec/viterbi.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace sih::fec {
namespace {

constexpr std::size_t constraint_length = 7;
constexpr std::size_t state_count = 1U << (constraint_length - 1U);
constexpr std::uint8_t polynomial_1 = 0171;
constexpr std::uint8_t polynomial_2 = 0133;
constexpr float negative_infinity = -std::numeric_limits<float>::infinity();
constexpr std::size_t cancellation_interval = 256;

struct Transition {
    std::uint8_t next_state;
    std::array<std::int8_t, 2> output_sign;
};

using TransitionTable = std::array<std::array<Transition, 2>, state_count>;

[[nodiscard]] const TransitionTable& transitions() {
    static const TransitionTable table = [] {
        TransitionTable result{};
        for (std::size_t state = 0; state < state_count; ++state) {
            for (std::size_t input = 0; input < 2; ++input) {
                const auto reg = static_cast<std::uint8_t>((input << (constraint_length - 1U)) | state);
                const auto out_1 = std::popcount(static_cast<unsigned>(reg & polynomial_1)) & 1U;
                const auto out_2 = std::popcount(static_cast<unsigned>(reg & polynomial_2)) & 1U;
                result[state][input] = Transition{
                    static_cast<std::uint8_t>((state >> 1U) | (input << (constraint_length - 2U))),
                    {static_cast<std::int8_t>(out_1 == 1U ? 1 : -1),
                     static_cast<std::int8_t>(out_2 == 1U ? 1 : -1)}};
            }
        }
        return result;
    }();
    return table;
}

}  // namespace

ViterbiResult decode_k7_r12_soft(
    const std::span<const float> llrs,
    const std::shared_ptr<core::CancellationToken>& cancellation,
    const std::shared_ptr<core::ProgressState>& progress) {
    if (llrs.size() % 2U != 0U) {
        throw std::invalid_argument("K=7 rate-1/2 Viterbi input requires an even number of LLRs");
    }
    if (!std::all_of(llrs.begin(), llrs.end(), [](const float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("Viterbi input LLRs must all be finite");
    }

    const auto symbol_count = llrs.size() / 2U;
    if (progress) {
        progress->update(0, symbol_count);
    }

    ViterbiResult result;
    result.residual_llrs = 0;
    if (symbol_count == 0) {
        return result;
    }
    if (cancellation && cancellation->is_cancelled()) {
        result.execution_status = core::ExecutionStatus::cancelled;
        result.residual_llrs = llrs.size();
        return result;
    }

    std::array<float, state_count> metrics{};
    std::array<float, state_count> next_metrics{};
    metrics.fill(negative_infinity);
    metrics[0] = 0.0F;
    std::vector<std::uint8_t> predecessor(symbol_count * state_count, 0);
    const auto& table = transitions();

    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        if ((symbol % cancellation_interval == 0U) && cancellation && cancellation->is_cancelled()) {
            result.execution_status = core::ExecutionStatus::cancelled;
            result.consumed_llrs = symbol * 2U;
            result.residual_llrs = llrs.size() - result.consumed_llrs;
            if (progress) {
                progress->update(symbol, symbol_count);
            }
            return result;
        }

        next_metrics.fill(negative_infinity);
        const float first = llrs[symbol * 2U];
        const float second = llrs[symbol * 2U + 1U];

        for (std::size_t state = 0; state < state_count; ++state) {
            if (!std::isfinite(metrics[state])) {
                continue;
            }
            for (std::size_t input = 0; input < 2U; ++input) {
                const auto& transition = table[state][input];
                const float branch = first * transition.output_sign[0] + second * transition.output_sign[1];
                const float candidate = metrics[state] + branch;
                if (candidate > next_metrics[transition.next_state]) {
                    next_metrics[transition.next_state] = candidate;
                    predecessor[symbol * state_count + transition.next_state] = static_cast<std::uint8_t>(state);
                }
            }
        }
        metrics = next_metrics;
        if (progress && ((symbol + 1U) % cancellation_interval == 0U)) {
            progress->update(symbol + 1U, symbol_count);
        }
    }

    std::array<float, state_count> sorted = metrics;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    result.path_metric_margin = sorted[0] - sorted[1];

    auto best = static_cast<std::uint8_t>(
        std::distance(metrics.begin(), std::max_element(metrics.begin(), metrics.end())));
    result.decoded_bits->resize(symbol_count);
    for (std::size_t symbol = symbol_count; symbol-- > 0U;) {
        (*result.decoded_bits)[symbol] = static_cast<std::uint8_t>(best >> (constraint_length - 2U));
        best = predecessor[symbol * state_count + best];
    }

    result.consumed_llrs = llrs.size();
    if (progress) {
        progress->update(symbol_count, symbol_count);
    }
    return result;
}

}  // namespace sih::fec
