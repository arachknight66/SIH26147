#include "sih/bitstream/bitstream.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {
bool require(const bool condition, const char* message) { if (!condition) std::cerr << message << '\n'; return condition; }
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

    LdpcMatrix matrix; matrix.variable_count = 3U; matrix.checks = {{0U, 1U}, {1U, 2U}};
    const std::vector<float> ldpc_llrs{-4.0F, -3.0F, -2.0F};
    const auto ldpc = decode_ldpc_normalized_min_sum(ldpc_llrs, matrix);
    passed &= require(ldpc.converged && ldpc.syndrome_weight == 0U, "LDPC normalized min-sum did not validate a zero-syndrome word");
    return passed ? 0 : 1;
}
