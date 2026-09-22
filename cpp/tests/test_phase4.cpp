#include "sih/receiver/receiver.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
constexpr double pi = 3.14159265358979323846;
bool require(bool condition, const char* message) { if (!condition) std::cerr << message << '\n'; return condition; }
}

int main() {
    constexpr std::size_t symbol_count = 512U;
    constexpr std::size_t sps = 4U;
    constexpr std::size_t acquisition = 8U;
    std::vector<std::complex<float>> samples;
    std::vector<std::uint8_t> expected;
    const std::vector<std::complex<double>> points{
        {1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)},
        {-1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)},
        {-1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0)},
        {1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0)}};
    const std::vector<std::vector<std::uint8_t>> labels{{1,1},{0,1},{0,0},{1,0}};
    for (std::size_t symbol = 0; symbol < symbol_count; ++symbol) {
        const auto choice = (symbol * 13U + 3U) % 4U;
        if (symbol >= acquisition) expected.insert(expected.end(), labels[choice].begin(), labels[choice].end());
        for (std::size_t sample = 0; sample < sps; ++sample) {
            const auto index = symbol * sps + sample;
            samples.emplace_back(points[choice] * std::polar(1.0, 2.0 * pi * 0.025 * index));
        }
    }
    sih::receiver::ReceiverConfig config;
    config.modulation = "QPSK";
    config.samples_per_symbol = sps;
    config.phase_reference_radians = 0.0;
    config.acquisition_symbols = acquisition;
    const auto direct = sih::receiver::demodulate(samples, config);
    bool passed = true;
    passed &= require(direct.acquisition_status == sih::receiver::AcquisitionStatus::locked, "QPSK receiver did not lock");
    passed &= require(*direct.hard_bits == expected, "noiseless QPSK bits were not exact after acquisition");
    passed &= require(direct.mapping_status == sih::core::EvidenceStatus::verified, "phase reference did not verify mapping");

    sih::receiver::ReceiverSession session(config);
    session.process(std::span(samples).first(17U));
    session.process(std::span(samples).subspan(17U, 701U));
    session.process(std::span(samples).subspan(718U));
    const auto chunked = session.flush();
    passed &= require(*chunked.hard_bits == *direct.hard_bits, "receiver chunk boundaries changed hard bits");
    passed &= require(*chunked.soft_llrs == *direct.soft_llrs, "receiver chunk boundaries changed LLRs");
    return passed ? 0 : 1;
}
