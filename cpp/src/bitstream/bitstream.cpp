#include "sih/bitstream/bitstream.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace sih::bitstream {
namespace {

[[nodiscard]] core::Diagnostic diagnostic(const core::Severity severity, std::string code,
                                          std::string message, std::string evidence = {}) {
    return {severity, std::move(code), std::move(message), std::move(evidence)};
}

[[nodiscard]] std::vector<std::size_t> block_permutation(const InterleaverConfig& config) {
    if (config.rows == 0U || config.columns == 0U) throw std::invalid_argument("block interleaver requires non-zero rows and columns");
    const auto total = config.rows * config.columns;
    std::vector<std::size_t> permutation(total);
    for (std::size_t row = 0; row < config.rows; ++row) {
        for (std::size_t column = 0; column < config.columns; ++column) {
            if (config.read_by_row) permutation[row * config.columns + column] = column * config.rows + row;
            else permutation[column * config.rows + row] = row * config.columns + column;
        }
    }
    return permutation;
}

[[nodiscard]] std::vector<std::size_t> diagonal_permutation(const InterleaverConfig& config) {
    if (config.rows == 0U || config.columns == 0U) throw std::invalid_argument("diagonal interleaver requires non-zero rows and columns");
    const auto total = config.rows * config.columns;
    std::vector<std::size_t> permutation(total);
    const auto slope = std::max<std::size_t>(1U, config.delay);
    for (std::size_t row = 0; row < config.rows; ++row) {
        for (std::size_t column = 0; column < config.columns; ++column) {
            const auto shifted = (column + row * slope) % config.columns;
            permutation[row * config.columns + column] = row * config.columns + shifted;
        }
    }
    return permutation;
}

[[nodiscard]] std::uint64_t next_random(std::uint64_t& state) {
    // xorshift64*: deterministic across compilers and platforms.
    state ^= state >> 12U;
    state ^= state << 25U;
    state ^= state >> 27U;
    return state * 2685821657736338717ULL;
}

[[nodiscard]] std::vector<std::size_t> seeded_permutation(const std::size_t length, const std::uint64_t seed) {
    std::vector<std::size_t> permutation(length);
    std::iota(permutation.begin(), permutation.end(), 0U);
    std::uint64_t state = seed == 0U ? 0x9e3779b97f4a7c15ULL : seed;
    for (std::size_t index = length; index > 1U; --index) {
        const auto other = static_cast<std::size_t>(next_random(state) % index);
        std::swap(permutation[index - 1U], permutation[other]);
    }
    return permutation;
}

[[nodiscard]] std::vector<std::size_t> convolutional_permutation(const InterleaverConfig& config) {
    if (config.branches == 0U || config.delay == 0U) throw std::invalid_argument("convolutional interleaver requires non-zero branches and delay");
    // A bounded, block-local commutator representation.  The span contains each
    // branch's delay-line period and avoids inventing a stream start state.
    const auto span = config.branches * config.delay;
    std::vector<std::size_t> permutation(span);
    for (std::size_t branch = 0; branch < config.branches; ++branch) {
        for (std::size_t depth = 0; depth < config.delay; ++depth) {
            const auto input = depth * config.branches + branch;
            const auto output = branch * config.delay + ((depth + branch) % config.delay);
            permutation[output] = input;
        }
    }
    return permutation;
}

[[nodiscard]] std::vector<std::size_t> permutation_for(const InterleaverConfig& config, const std::size_t length) {
    switch (config.family) {
        case InterleaverFamily::block: return block_permutation(config);
        case InterleaverFamily::diagonal: return diagonal_permutation(config);
        case InterleaverFamily::convolutional: return convolutional_permutation(config);
        case InterleaverFamily::pseudo_random:
            if (!config.permutation.empty()) return config.permutation;
            return seeded_permutation(length, config.seed);
        case InterleaverFamily::none: return {};
    }
    throw std::invalid_argument("unknown interleaver family");
}

void validate_permutation(const std::span<const std::size_t> permutation, const std::size_t expected) {
    if (permutation.size() != expected) throw std::invalid_argument("interleaver permutation size does not match its block span");
    std::vector<bool> seen(expected, false);
    for (const auto value : permutation) {
        if (value >= expected || seen[value]) throw std::invalid_argument("interleaver permutation must contain each index exactly once");
        seen[value] = true;
    }
}

template <typename Value>
[[nodiscard]] std::vector<Value> apply_forward(const std::span<const Value> values,
                                                const std::span<const std::size_t> permutation,
                                                const std::size_t block_span) {
    std::vector<Value> output(values.begin(), values.end());
    if (block_span == 0U) return output;
    const auto complete = values.size() / block_span;
    for (std::size_t block = 0; block < complete; ++block) {
        const auto base = block * block_span;
        for (std::size_t index = 0; index < block_span; ++index) output[base + index] = values[base + permutation[index]];
    }
    return output;
}

template <typename Value>
[[nodiscard]] std::vector<Value> apply_inverse(const std::span<const Value> values,
                                                const std::span<const std::size_t> permutation,
                                                const std::size_t block_span) {
    std::vector<Value> output(values.begin(), values.end());
    if (block_span == 0U) return output;
    const auto complete = values.size() / block_span;
    for (std::size_t block = 0; block < complete; ++block) {
        const auto base = block * block_span;
        for (std::size_t index = 0; index < block_span; ++index) output[base + permutation[index]] = values[base + index];
    }
    return output;
}

class Gf256 final {
public:
    Gf256() {
        std::uint16_t value = 1U;
        for (std::size_t index = 0; index < 255U; ++index) {
            exp_[index] = static_cast<std::uint8_t>(value);
            log_[value] = static_cast<std::uint8_t>(index);
            value <<= 1U;
            if ((value & 0x100U) != 0U) value ^= 0x11dU;
        }
        for (std::size_t index = 255U; index < exp_.size(); ++index) exp_[index] = exp_[index - 255U];
    }
    [[nodiscard]] std::uint8_t add(const std::uint8_t left, const std::uint8_t right) const { return left ^ right; }
    [[nodiscard]] std::uint8_t multiply(const std::uint8_t left, const std::uint8_t right) const {
        return left == 0U || right == 0U ? 0U : exp_[log_[left] + log_[right]];
    }
    [[nodiscard]] std::uint8_t divide(const std::uint8_t left, const std::uint8_t right) const {
        if (right == 0U) throw std::domain_error("GF(256) division by zero");
        if (left == 0U) return 0U;
        auto index = static_cast<int>(log_[left]) - static_cast<int>(log_[right]);
        if (index < 0) index += 255;
        return exp_[static_cast<std::size_t>(index)];
    }
    [[nodiscard]] std::uint8_t power_alpha(const std::size_t exponent) const { return exp_[exponent % 255U]; }
    [[nodiscard]] std::uint8_t inverse(const std::uint8_t value) const {
        if (value == 0U) throw std::domain_error("GF(256) inverse of zero");
        return exp_[255U - log_[value]];
    }
private:
    std::array<std::uint8_t, 510U> exp_{};
    std::array<std::uint8_t, 256U> log_{};
};

[[nodiscard]] const Gf256& gf() { static const Gf256 field; return field; }

[[nodiscard]] std::vector<std::uint8_t> generator(const ReedSolomonConfig& config) {
    std::vector<std::uint8_t> polynomial{1U};
    for (std::size_t root = 0; root < config.n - config.k; ++root) {
        std::vector<std::uint8_t> next(polynomial.size() + 1U, 0U);
        for (std::size_t index = 0; index < polynomial.size(); ++index) {
            next[index] ^= polynomial[index];
            next[index + 1U] ^= gf().multiply(polynomial[index], gf().power_alpha(root));
        }
        polynomial = std::move(next);
    }
    return polynomial;
}

[[nodiscard]] std::vector<std::uint8_t> syndromes(const std::span<const std::uint8_t> word, const std::size_t count) {
    std::vector<std::uint8_t> values(count, 0U);
    for (std::size_t root = 0; root < count; ++root) {
        std::uint8_t total = 0U;
        const auto point = gf().power_alpha(root);
        for (const auto value : word) total = gf().multiply(total, point) ^ value;
        values[root] = total;
    }
    return values;
}

[[nodiscard]] std::vector<std::uint8_t> berlekamp_massey(const std::span<const std::uint8_t> syndrome) {
    std::vector<std::uint8_t> locator{1U}, previous{1U};
    std::size_t degree = 0U, shift = 1U;
    std::uint8_t previous_discrepancy = 1U;
    for (std::size_t index = 0; index < syndrome.size(); ++index) {
        std::uint8_t discrepancy = syndrome[index];
        for (std::size_t order = 1U; order <= degree; ++order) discrepancy ^= gf().multiply(locator[order], syndrome[index - order]);
        if (discrepancy == 0U) { ++shift; continue; }
        const auto saved = locator;
        const auto scale = gf().divide(discrepancy, previous_discrepancy);
        if (locator.size() < previous.size() + shift) locator.resize(previous.size() + shift, 0U);
        for (std::size_t item = 0; item < previous.size(); ++item) locator[item + shift] ^= gf().multiply(scale, previous[item]);
        if (2U * degree <= index) {
            degree = index + 1U - degree;
            previous = std::move(saved);
            previous_discrepancy = discrepancy;
            shift = 1U;
        } else ++shift;
    }
    locator.resize(degree + 1U);
    return locator;
}

[[nodiscard]] std::uint8_t evaluate_ascending(const std::span<const std::uint8_t> polynomial, const std::uint8_t point) {
    std::uint8_t total = 0U;
    for (auto iterator = polynomial.rbegin(); iterator != polynomial.rend(); ++iterator) total = gf().multiply(total, point) ^ *iterator;
    return total;
}

[[nodiscard]] bool solve_vandermonde(std::vector<std::vector<std::uint8_t>> matrix,
                                     std::vector<std::uint8_t> values, std::vector<std::uint8_t>& solution) {
    const auto size = matrix.size();
    for (std::size_t pivot = 0; pivot < size; ++pivot) {
        std::size_t source = pivot;
        while (source < size && matrix[source][pivot] == 0U) ++source;
        if (source == size) return false;
        if (source != pivot) { std::swap(matrix[source], matrix[pivot]); std::swap(values[source], values[pivot]); }
        const auto inverse = gf().inverse(matrix[pivot][pivot]);
        for (std::size_t column = pivot; column < size; ++column) matrix[pivot][column] = gf().multiply(matrix[pivot][column], inverse);
        values[pivot] = gf().multiply(values[pivot], inverse);
        for (std::size_t row = 0; row < size; ++row) {
            if (row == pivot || matrix[row][pivot] == 0U) continue;
            const auto scale = matrix[row][pivot];
            for (std::size_t column = pivot; column < size; ++column) matrix[row][column] ^= gf().multiply(scale, matrix[pivot][column]);
            values[row] ^= gf().multiply(scale, values[pivot]);
        }
    }
    solution = std::move(values);
    return true;
}

[[nodiscard]] std::uint32_t reverse_bits(std::uint32_t value, const std::uint8_t width) {
    std::uint32_t result = 0U;
    for (std::uint8_t bit = 0U; bit < width; ++bit) { result = (result << 1U) | (value & 1U); value >>= 1U; }
    return result;
}

}  // namespace

DeinterleaveResult deinterleave(const std::span<const std::uint8_t> bits, const std::span<const float> llrs,
                                const InterleaverConfig& config) {
    if (!llrs.empty() && bits.size() != llrs.size()) throw std::invalid_argument("bit and LLR streams must have equal lengths when LLRs are supplied");
    if (!std::all_of(bits.begin(), bits.end(), [](const std::uint8_t value) { return value <= 1U; })) throw std::invalid_argument("bitstream values must be 0 or 1");
    if (!std::all_of(llrs.begin(), llrs.end(), [](const float value) { return std::isfinite(value); })) throw std::invalid_argument("LLRs must be finite");
    DeinterleaveResult result;
    if (config.family == InterleaverFamily::none) { *result.bits = {bits.begin(), bits.end()}; *result.llrs = {llrs.begin(), llrs.end()}; return result; }
    const auto permutation = permutation_for(config, config.family == InterleaverFamily::pseudo_random ? bits.size() : 0U);
    validate_permutation(permutation, permutation.size());
    *result.bits = apply_inverse(bits, permutation, permutation.size());
    if (!llrs.empty()) *result.llrs = apply_inverse(llrs, permutation, permutation.size());
    result.transformed_bits = bits.size() / permutation.size() * permutation.size();
    result.residual_bits = bits.size() - result.transformed_bits;
    if (result.residual_bits != 0U) result.diagnostics.push_back(diagnostic(core::Severity::warning, "INTERLEAVER_RESIDUAL_BITS", "Input ends with an incomplete interleaver block; residual bits were preserved unchanged.", std::to_string(result.residual_bits)));
    return result;
}

std::vector<std::uint8_t> interleave_bits(const std::span<const std::uint8_t> bits, const InterleaverConfig& config) {
    if (config.family == InterleaverFamily::none) return {bits.begin(), bits.end()};
    const auto permutation = permutation_for(config, config.family == InterleaverFamily::pseudo_random ? bits.size() : 0U);
    validate_permutation(permutation, permutation.size());
    return apply_forward(bits, permutation, permutation.size());
}

std::vector<float> interleave_llrs(const std::span<const float> llrs, const InterleaverConfig& config) {
    if (!std::all_of(llrs.begin(), llrs.end(), [](const float value) { return std::isfinite(value); })) throw std::invalid_argument("LLRs must be finite");
    if (config.family == InterleaverFamily::none) return {llrs.begin(), llrs.end()};
    const auto permutation = permutation_for(config, config.family == InterleaverFamily::pseudo_random ? llrs.size() : 0U);
    validate_permutation(permutation, permutation.size());
    return apply_forward(llrs, permutation, permutation.size());
}

std::vector<CorrelationMatch> correlate(const std::span<const std::uint8_t> bits, const std::span<const float> llrs,
                                        const std::span<const CorrelationPattern> patterns, const float maximum_hamming_fraction) {
    if (!llrs.empty() && bits.size() != llrs.size()) throw std::invalid_argument("bit and LLR streams must have equal lengths when LLRs are supplied");
    std::vector<CorrelationMatch> matches;
    float median = 0.0F;
    if (!llrs.empty()) { std::vector<float> values; values.reserve(llrs.size()); for (const auto value : llrs) values.push_back(std::abs(value)); std::nth_element(values.begin(), values.begin() + values.size() / 2U, values.end()); median = values[values.size() / 2U]; }
    for (const auto& pattern : patterns) {
        if (pattern.bits.empty() || pattern.bits.size() > bits.size()) continue;
        const auto maximum = pattern.bits.size() < 16U ? 0U : static_cast<std::size_t>(std::floor(pattern.bits.size() * maximum_hamming_fraction));
        for (std::size_t offset = 0; offset + pattern.bits.size() <= bits.size(); ++offset) {
            std::size_t distance = 0U; float local = 0.0F;
            for (std::size_t index = 0; index < pattern.bits.size(); ++index) { distance += bits[offset + index] != pattern.bits[index]; if (!llrs.empty()) local += std::abs(llrs[offset + index]); }
            if (distance > maximum) continue;
            const auto score = (1.0F - static_cast<float>(distance) / static_cast<float>(maximum + 1U)) * (median > 1.0e-9F ? std::min(1.0F, local / (median * pattern.bits.size())) : 0.5F);
            matches.push_back({pattern.name, offset, distance, score, false});
        }
    }
    for (auto& match : matches) {
        for (const auto& other : matches) {
            if (match.pattern_name == other.pattern_name && other.bit_offset > match.bit_offset && other.bit_offset - match.bit_offset >= 16U) {
                const auto period = other.bit_offset - match.bit_offset;
                const auto third = std::find_if(matches.begin(), matches.end(), [&](const CorrelationMatch& value) { return value.pattern_name == match.pattern_name && value.bit_offset == other.bit_offset + period; });
                if (third != matches.end()) { match.periodic = true; match.confidence = std::min(1.0F, match.confidence + 0.3F); break; }
            }
        }
        if (match.periodic) continue;
        for (const auto& previous : matches) {
            if (previous.pattern_name != match.pattern_name || previous.bit_offset >= match.bit_offset) continue;
            const auto period = match.bit_offset - previous.bit_offset;
            const auto following = std::find_if(matches.begin(), matches.end(), [&](const CorrelationMatch& value) { return value.pattern_name == match.pattern_name && value.bit_offset == match.bit_offset + period; });
            if (following != matches.end()) { match.periodic = true; match.confidence = std::min(1.0F, match.confidence + 0.3F); break; }
        }
    }
    std::sort(matches.begin(), matches.end(), [](const auto& left, const auto& right) { return left.confidence > right.confidence; });
    return matches;
}

std::uint32_t crc_bits(const std::span<const std::uint8_t> bits, const CrcConfig& config) {
    if (config.width == 0U || config.width > 32U) throw std::invalid_argument("CRC width must be between 1 and 32");
    const auto mask = config.width == 32U ? 0xffffffffU : ((1U << config.width) - 1U);
    std::uint32_t state = config.initial & mask;
    for (const auto raw_bit : bits) {
        if (raw_bit > 1U) throw std::invalid_argument("CRC input must contain bits");
        const auto bit = static_cast<std::uint32_t>(raw_bit);
        if (config.reflect_input) {
            const auto feedback = (state & 1U) ^ bit;
            state >>= 1U;
            if (feedback != 0U) state ^= reverse_bits(config.polynomial, config.width);
        } else {
            const auto feedback = ((state >> (config.width - 1U)) & 1U) ^ bit;
            state = (state << 1U) & mask;
            if (feedback != 0U) state ^= config.polynomial;
        }
    }
    if (config.reflect_output != config.reflect_input) state = reverse_bits(state, config.width);
    return (state ^ config.xor_output) & mask;
}

std::vector<std::uint8_t> encode_reed_solomon(const std::span<const std::uint8_t> message, const ReedSolomonConfig& config) {
    if (config.k == 0U || config.n <= config.k || config.n > 255U) throw std::invalid_argument("RS requires 0 < k < n <= 255");
    if (message.size() != config.k) throw std::invalid_argument("RS encoder requires exactly k message bytes");
    auto output = std::vector<std::uint8_t>(message.begin(), message.end());
    output.resize(config.n, 0U);
    const auto polynomial = generator(config);
    for (std::size_t index = 0; index < config.k; ++index) {
        const auto coefficient = output[index];
        if (coefficient == 0U) continue;
        for (std::size_t term = 1U; term < polynomial.size(); ++term) output[index + term] ^= gf().multiply(polynomial[term], coefficient);
    }
    std::copy(message.begin(), message.end(), output.begin());
    return output;
}

ReedSolomonResult decode_reed_solomon(const std::span<const std::uint8_t> bytes, const ReedSolomonConfig& config) {
    if (config.k == 0U || config.n <= config.k || config.n > 255U) throw std::invalid_argument("RS requires 0 < k < n <= 255");
    ReedSolomonResult result;
    const auto full_blocks = bytes.size() / config.n;
    result.consumed_bytes = full_blocks * config.n;
    result.residual_bytes = bytes.size() - result.consumed_bytes;
    if (result.residual_bytes != 0U) result.diagnostics.push_back(diagnostic(core::Severity::warning, "RS_RESIDUAL_BYTES", "Trailing bytes do not form a complete RS codeword and were not decoded.", std::to_string(result.residual_bytes)));
    if (full_blocks == 0U) { result.diagnostics.push_back(diagnostic(core::Severity::warning, "RS_INSUFFICIENT_DATA", "No complete RS codeword was available.")); return result; }
    const auto parity = config.n - config.k;
    bool all_valid = true;
    for (std::size_t block = 0; block < full_blocks; ++block) {
        std::vector<std::uint8_t> word(bytes.begin() + static_cast<std::ptrdiff_t>(block * config.n), bytes.begin() + static_cast<std::ptrdiff_t>((block + 1U) * config.n));
        const auto syndrome = syndromes(word, parity);
        result.syndrome_weight += static_cast<std::size_t>(std::count_if(syndrome.begin(), syndrome.end(), [](const auto value) { return value != 0U; }));
        if (std::all_of(syndrome.begin(), syndrome.end(), [](const auto value) { return value == 0U; })) { result.decoded_bytes->insert(result.decoded_bytes->end(), word.begin(), word.begin() + static_cast<std::ptrdiff_t>(config.k)); continue; }
        const auto locator = berlekamp_massey(syndrome);
        const auto errors = locator.size() - 1U;
        if (errors == 0U || errors > parity / 2U) {
            all_valid = false;
            result.diagnostics.push_back(diagnostic(core::Severity::error, "RS_CORRECTION_CAPACITY_EXCEEDED", "Syndrome requires more errors than this RS profile can correct.", std::to_string(errors)));
            continue;
        }
        std::vector<std::size_t> locations;
        std::vector<std::uint8_t> locators;
        for (std::size_t exponent = 0; exponent < config.n; ++exponent) {
            const auto x = gf().power_alpha((255U - exponent) % 255U);
            if (evaluate_ascending(locator, x) == 0U) { locations.push_back(config.n - 1U - exponent); locators.push_back(gf().power_alpha(exponent)); }
        }
        if (locations.size() != errors) {
            all_valid = false;
            result.diagnostics.push_back(diagnostic(core::Severity::error, "RS_ROOT_MISMATCH", "Error locator root count did not match its degree."));
            continue;
        }
        std::vector<std::vector<std::uint8_t>> matrix(errors, std::vector<std::uint8_t>(errors, 0U));
        for (std::size_t row = 0; row < errors; ++row) for (std::size_t column = 0; column < errors; ++column) matrix[row][column] = row == 0U ? 1U : gf().multiply(matrix[row - 1U][column], locators[column]);
        std::vector<std::uint8_t> magnitudes;
        if (!solve_vandermonde(std::move(matrix), std::vector<std::uint8_t>(syndrome.begin(), syndrome.begin() + static_cast<std::ptrdiff_t>(errors)), magnitudes)) {
            all_valid = false;
            result.diagnostics.push_back(diagnostic(core::Severity::error, "RS_MAGNITUDE_SOLVE_FAILED", "Could not solve the RS error-magnitude system."));
            continue;
        }
        for (std::size_t index = 0; index < errors; ++index) word[locations[index]] ^= magnitudes[index];
        const auto verified = syndromes(word, parity);
        if (!std::all_of(verified.begin(), verified.end(), [](const auto value) { return value == 0U; })) {
            all_valid = false;
            result.diagnostics.push_back(diagnostic(core::Severity::error, "RS_DECODE_FAILED", "Syndromes remained non-zero after correction."));
            continue;
        }
        result.corrected_symbols += errors;
        result.decoded_bytes->insert(result.decoded_bytes->end(), word.begin(), word.begin() + static_cast<std::ptrdiff_t>(config.k));
    }
    result.success = all_valid && result.residual_bytes == 0U;
    return result;
}

LdpcResult decode_ldpc_normalized_min_sum(const std::span<const float> llrs, const LdpcMatrix& matrix, const LdpcConfig& config) {
    if (matrix.variable_count == 0U || llrs.size() != matrix.variable_count) throw std::invalid_argument("LDPC LLR count must equal matrix variable count");
    if (config.maximum_iterations == 0U || config.normalization <= 0.0F || config.normalization > 1.0F) throw std::invalid_argument("invalid LDPC normalized min-sum configuration");
    std::vector<std::vector<std::size_t>> variable_checks(matrix.variable_count);
    for (std::size_t check = 0; check < matrix.checks.size(); ++check) for (const auto variable : matrix.checks[check]) { if (variable >= matrix.variable_count) throw std::invalid_argument("LDPC check index exceeds variable count"); variable_checks[variable].push_back(check); }
    std::vector<std::vector<float>> check_to_variable(matrix.checks.size());
    std::vector<std::vector<float>> variable_to_check(matrix.checks.size());
    for (std::size_t check = 0; check < matrix.checks.size(); ++check) { check_to_variable[check].assign(matrix.checks[check].size(), 0.0F); variable_to_check[check].assign(matrix.checks[check].size(), 0.0F); }
    LdpcResult result;
    std::vector<float> posterior(llrs.begin(), llrs.end());
    for (std::size_t iteration = 0; iteration < config.maximum_iterations; ++iteration) {
        for (std::size_t check = 0; check < matrix.checks.size(); ++check) for (std::size_t edge = 0; edge < matrix.checks[check].size(); ++edge) { const auto variable = matrix.checks[check][edge]; float sum = llrs[variable]; for (const auto adjacent : variable_checks[variable]) { const auto position = static_cast<std::size_t>(std::find(matrix.checks[adjacent].begin(), matrix.checks[adjacent].end(), variable) - matrix.checks[adjacent].begin()); if (adjacent != check) sum += check_to_variable[adjacent][position]; } variable_to_check[check][edge] = sum; }
        for (std::size_t check = 0; check < matrix.checks.size(); ++check) for (std::size_t edge = 0; edge < matrix.checks[check].size(); ++edge) { float minimum = std::numeric_limits<float>::infinity(); float sign = 1.0F; for (std::size_t other = 0; other < matrix.checks[check].size(); ++other) if (other != edge) { const auto value = variable_to_check[check][other]; minimum = std::min(minimum, std::abs(value)); if (value < 0.0F) sign = -sign; } check_to_variable[check][edge] = sign * config.normalization * minimum; }
        for (std::size_t variable = 0; variable < matrix.variable_count; ++variable) { posterior[variable] = llrs[variable]; for (const auto check : variable_checks[variable]) { const auto edge = static_cast<std::size_t>(std::find(matrix.checks[check].begin(), matrix.checks[check].end(), variable) - matrix.checks[check].begin()); posterior[variable] += check_to_variable[check][edge]; } (*result.decoded_bits).push_back(posterior[variable] > 0.0F ? 1U : 0U); }
        result.syndrome_weight = 0U;
        for (const auto& check : matrix.checks) { std::uint8_t parity = 0U; for (const auto variable : check) parity ^= (*result.decoded_bits)[variable]; result.syndrome_weight += parity; }
        result.iterations = iteration + 1U;
        if (result.syndrome_weight == 0U) { result.converged = true; return result; }
        result.decoded_bits->clear();
    }
    result.diagnostics.push_back(diagnostic(core::Severity::warning, "LDPC_MAX_ITERATIONS", "LDPC normalized min-sum did not reach a zero syndrome within its configured iteration budget.", std::to_string(result.syndrome_weight)));
    for (const auto value : posterior) result.decoded_bits->push_back(value > 0.0F ? 1U : 0U);
    return result;
}

LdpcMatrix load_alist(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open LDPC alist matrix: " + path);
    std::size_t variables = 0U, checks = 0U, maximum_column = 0U, maximum_row = 0U;
    input >> variables >> checks >> maximum_column >> maximum_row;
    if (!input || variables == 0U || checks == 0U) throw std::invalid_argument("invalid LDPC alist header");
    std::vector<std::size_t> column_weights(variables), row_weights(checks);
    for (auto& value : column_weights) input >> value;
    for (auto& value : row_weights) input >> value;
    LdpcMatrix result; result.variable_count = variables; result.checks.assign(checks, {});
    for (std::size_t variable = 0; variable < variables; ++variable) for (std::size_t index = 0; index < maximum_column; ++index) { std::size_t one_based = 0U; input >> one_based; if (!input) throw std::invalid_argument("truncated LDPC alist column section"); if (one_based != 0U) { if (one_based > checks) throw std::invalid_argument("LDPC alist check index out of range"); result.checks[one_based - 1U].push_back(variable); } }
    return result;
}

}  // namespace sih::bitstream
