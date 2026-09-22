#pragma once

#include "sih/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sih::bitstream {

enum class InterleaverFamily : std::uint8_t { none, block, convolutional, diagonal, pseudo_random };

struct InterleaverConfig {
    InterleaverFamily family{InterleaverFamily::none};
    std::size_t rows{0};
    std::size_t columns{0};
    bool read_by_row{true};
    std::size_t branches{0};
    std::size_t delay{0};
    std::uint64_t seed{0};
    std::vector<std::size_t> permutation;
};

struct DeinterleaveResult {
    std::shared_ptr<std::vector<std::uint8_t>> bits{std::make_shared<std::vector<std::uint8_t>>()};
    std::shared_ptr<std::vector<float>> llrs{std::make_shared<std::vector<float>>()};
    std::size_t transformed_bits{0};
    std::size_t residual_bits{0};
    std::vector<core::Diagnostic> diagnostics;
};

struct CorrelationPattern {
    std::string name;
    std::vector<std::uint8_t> bits;
};

struct CorrelationMatch {
    std::string pattern_name;
    std::size_t bit_offset{0};
    std::size_t hamming_distance{0};
    float confidence{0.0F};
    bool periodic{false};
};

struct CrcConfig {
    std::string name;
    std::uint8_t width{0};
    std::uint32_t polynomial{0};
    std::uint32_t initial{0};
    bool reflect_input{false};
    bool reflect_output{false};
    std::uint32_t xor_output{0};
};

struct ReedSolomonConfig {
    std::size_t n{255};
    std::size_t k{223};
};

struct ReedSolomonResult {
    std::shared_ptr<std::vector<std::uint8_t>> decoded_bytes{std::make_shared<std::vector<std::uint8_t>>()};
    bool success{false};
    std::size_t corrected_symbols{0};
    std::size_t consumed_bytes{0};
    std::size_t residual_bytes{0};
    std::size_t syndrome_weight{0};
    std::vector<core::Diagnostic> diagnostics;
};

struct LdpcMatrix {
    std::size_t variable_count{0};
    std::vector<std::vector<std::size_t>> checks;
};

struct LdpcConfig {
    std::size_t maximum_iterations{50};
    float normalization{0.8F};
};

struct LdpcResult {
    std::shared_ptr<std::vector<std::uint8_t>> decoded_bits{std::make_shared<std::vector<std::uint8_t>>()};
    bool converged{false};
    std::size_t iterations{0};
    std::size_t syndrome_weight{0};
    std::vector<core::Diagnostic> diagnostics;
};

[[nodiscard]] DeinterleaveResult deinterleave(
    std::span<const std::uint8_t> bits, std::span<const float> llrs, const InterleaverConfig& config);
[[nodiscard]] std::vector<std::uint8_t> interleave_bits(
    std::span<const std::uint8_t> bits, const InterleaverConfig& config);
[[nodiscard]] std::vector<float> interleave_llrs(
    std::span<const float> llrs, const InterleaverConfig& config);
[[nodiscard]] std::vector<CorrelationMatch> correlate(
    std::span<const std::uint8_t> bits, std::span<const float> llrs,
    std::span<const CorrelationPattern> patterns, float maximum_hamming_fraction = 0.15F);
[[nodiscard]] std::uint32_t crc_bits(std::span<const std::uint8_t> bits, const CrcConfig& config);
[[nodiscard]] ReedSolomonResult decode_reed_solomon(
    std::span<const std::uint8_t> bytes, const ReedSolomonConfig& config);
[[nodiscard]] std::vector<std::uint8_t> encode_reed_solomon(
    std::span<const std::uint8_t> message, const ReedSolomonConfig& config);
[[nodiscard]] LdpcResult decode_ldpc_normalized_min_sum(
    std::span<const float> llrs, const LdpcMatrix& matrix, const LdpcConfig& config = {});
[[nodiscard]] LdpcMatrix load_alist(const std::string& path);

}  // namespace sih::bitstream
