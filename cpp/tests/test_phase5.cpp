#include "sih/bitstream/bitstream.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {
bool require(const bool condition, const char* message) { if (!condition) std::cerr << message << '\n'; return condition; }

// Test-only GF(256) evaluator.  It evaluates codewords emitted by the actual
// native encoder; no production generator implementation is duplicated here.
std::uint8_t gf_multiply(const std::uint8_t left, const std::uint8_t right) {
    std::uint8_t a = left, b = right, product = 0U;
    while (b != 0U) { if ((b & 1U) != 0U) product ^= a; const bool high = (a & 0x80U) != 0U; a <<= 1U; if (high) a ^= 0x1dU; b >>= 1U; }
    return product;
}
std::uint8_t alpha_power(std::size_t exponent) { std::uint8_t value = 1U; while (exponent-- != 0U) value = gf_multiply(value, 2U); return value; }
std::uint8_t evaluate(const std::vector<std::uint8_t>& polynomial, const std::uint8_t x) { std::uint8_t value = 0U; for (const auto coefficient : polynomial) value = gf_multiply(value, x) ^ coefficient; return value; }
bool roots_hold_for_native_encoder(const sih::bitstream::ReedSolomonConfig& config) {
    const auto parity = config.n - config.k;
    // Systematic basis messages span every emitted codeword, so this checks the
    // actual native generator's BCH roots, not an independently recreated g(x).
    bool right_adjacent_nonroot = false, left_adjacent_nonroot = false;
    for (std::size_t basis = 0; basis < config.k; ++basis) {
        std::vector<std::uint8_t> message(config.k, 0U); message[basis] = 1U;
        const auto word = sih::bitstream::encode_reed_solomon(message, config);
        for (std::size_t root = 0; root < parity; ++root) if (evaluate(word, alpha_power(root)) != 0U) return false;
        right_adjacent_nonroot |= evaluate(word, alpha_power(parity)) != 0U;
        left_adjacent_nonroot |= evaluate(word, alpha_power(254U)) != 0U;
    }
    return right_adjacent_nonroot && left_adjacent_nonroot;
}
}

int main() {
    using namespace sih::bitstream;
    bool passed = true;
    std::vector<std::uint8_t> bits(96U);
    std::vector<float> llrs(96U);
    for (std::size_t index = 0; index < bits.size(); ++index) { bits[index] = static_cast<std::uint8_t>((index * 7U + 3U) & 1U); llrs[index] = bits[index] ? 3.0F : -3.0F; }
    InterleaverConfig block; block.family = InterleaverFamily::block; block.rows = 8U; block.columns = 12U; block.read_by_row = false;
    const auto interleaved = interleave_bits(bits, block);
    const auto restored = deinterleave(interleaved, llrs, block);
    passed &= require(*restored.bits == bits, "block deinterleaver did not invert the documented block permutation");
    InterleaverConfig pseudo; pseudo.family = InterleaverFamily::pseudo_random; pseudo.seed = 26147U;
    const auto shuffled = interleave_bits(bits, pseudo);
    const auto unshuffled = deinterleave(shuffled, llrs, pseudo);
    passed &= require(*unshuffled.bits == bits, "seeded pseudo-random deinterleaver did not invert its permutation");

    ReedSolomonConfig rs; rs.n = 15U; rs.k = 11U;
    std::vector<std::uint8_t> message(11U);
    for (std::size_t index = 0; index < message.size(); ++index) message[index] = static_cast<std::uint8_t>(index * 19U + 1U);
    auto word = encode_reed_solomon(message, rs);
    word[2] ^= 0x55U; word[9] ^= 0x13U;
    const auto decoded = decode_reed_solomon(word, rs);
    passed &= require(decoded.success, "RS decoder did not correct errors within its correction capacity");
    passed &= require(*decoded.decoded_bytes == message, "RS decoder output differs from the original message");
    word[0] ^= 0x77U;
    const auto rejected = decode_reed_solomon(word, rs);
    passed &= require(!rejected.success, "RS decoder accepted a codeword beyond correction capacity");

    for (const ReedSolomonConfig profile : {ReedSolomonConfig{255U, 223U}, ReedSolomonConfig{255U, 239U}, ReedSolomonConfig{15U, 11U}}) {
        passed &= require(roots_hold_for_native_encoder(profile), "native RS encoder codeword span did not vanish at all configured consecutive BCH roots");
    }

    LdpcMatrix matrix; matrix.variable_count = 3U; matrix.checks = {{0U, 1U}, {1U, 2U}};
    const std::vector<float> ldpc_llrs{-4.0F, -3.0F, -2.0F};
    const auto ldpc = decode_ldpc_normalized_min_sum(ldpc_llrs, matrix);
    passed &= require(ldpc.converged && ldpc.syndrome_weight == 0U, "LDPC normalized min-sum did not validate a zero-syndrome word");
    return passed ? 0 : 1;
}
