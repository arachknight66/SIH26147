#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sih::estimation {

enum class Validity : std::uint8_t { valid, unreliable, unavailable };
enum class InferenceStatus : std::uint8_t { candidate, ambiguous, unknown, unsupported };

struct Estimate {
    std::optional<double> value;
    std::optional<double> uncertainty;
    std::string unit;
    Validity validity{Validity::unavailable};
    std::string evidence;
};

struct RateCandidate {
    double symbol_rate{0.0};
    double samples_per_symbol{0.0};
    double score{0.0};
};

struct ModulationCandidate {
    std::string label;
    std::string family;
    double score{0.0};
    InferenceStatus status{InferenceStatus::candidate};
    double constellation_error{0.0};
    double temporal_consistency{0.0};
    std::vector<std::string> evidence;
    std::vector<std::string> contradictions;
};

struct AnalysisConfig {
    std::uint32_t schema_version{1};
    std::optional<double> sample_rate_hz;
    bool complex_input{true};
    double minimum_samples_per_symbol{2.0};
    double maximum_samples_per_symbol{32.0};
    std::size_t maximum_candidates{5};
    std::size_t temporal_windows{4};
    double unknown_threshold{0.55};
    double ambiguity_margin{0.06};
};

struct AnalysisResult {
    Estimate snr;
    Estimate occupied_bandwidth;
    Estimate carrier_offset;
    Estimate symbol_rate;
    std::vector<RateCandidate> rate_candidates;
    std::vector<ModulationCandidate> modulation_candidates;
    InferenceStatus status{InferenceStatus::unknown};
    double temporal_consistency{0.0};
    std::uint64_t processed_samples{0};
    std::vector<std::string> diagnostics;
};

[[nodiscard]] AnalysisResult analyze_window(
    std::span<const std::complex<float>> samples,
    const AnalysisConfig& config = {});

}  // namespace sih::estimation
