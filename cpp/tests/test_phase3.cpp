#include "sih/estimation/analysis.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr double pi = 3.14159265358979323846;

std::vector<std::complex<float>> make_psk(
    const unsigned order, const std::size_t symbols, const std::size_t sps,
    const double cfo, const double snr_db, const std::uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_int_distribution<unsigned> choice(0U, order - 1U);
    std::normal_distribution<double> gaussian(0.0, std::pow(10.0, -snr_db / 20.0) / std::sqrt(2.0));
    std::vector<std::complex<float>> result;
    result.reserve(symbols * sps);
    for (std::size_t symbol = 0; symbol < symbols; ++symbol) {
        const auto point = std::polar(1.0, 2.0 * pi * choice(random) / order);
        for (std::size_t sample = 0; sample < sps; ++sample) {
            const auto index = symbol * sps + sample;
            const auto value = point * std::polar(1.0, 2.0 * pi * cfo * index) +
                               std::complex<double>{gaussian(random), gaussian(random)};
            result.emplace_back(value);
        }
    }
    return result;
}

bool require(const bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}
}

int main() {
    bool passed = true;
    for (const auto& [order, label] : std::vector<std::pair<unsigned, std::string>>{
             {2U, "BPSK"}, {4U, "QPSK"}, {8U, "8PSK"}}) {
        const auto samples = make_psk(order, 2500U, 4U, 0.025, 18.0, order);
        const auto result = sih::estimation::analyze_window(samples);
        passed &= require(result.symbol_rate.value.has_value(), "symbol rate unavailable");
        passed &= require(std::abs(*result.symbol_rate.value - 0.25) <= 0.005,
                          "symbol rate exceeded two-percent tolerance");
        passed &= require(!result.modulation_candidates.empty(), "no modulation candidates");
        passed &= require(result.modulation_candidates.front().label == label,
                          "wrong PSK modulation winner");
        passed &= require(result.status == sih::estimation::InferenceStatus::candidate,
                          "supported signal was not a candidate");
    }

    const std::vector<std::complex<float>> silent(1024U);
    const auto silence = sih::estimation::analyze_window(silent);
    passed &= require(silence.status == sih::estimation::InferenceStatus::unknown,
                      "silence must be unknown");
    passed &= require(silence.modulation_candidates.empty(),
                      "silence must not produce a modulation candidate");
    return passed ? 0 : 1;
}
