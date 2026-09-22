#pragma once

#include "sih/core/types.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sih::receiver {

enum class AcquisitionStatus : std::uint8_t { locked, unlocked, unsupported, insufficient_data };

struct ReceiverConfig {
    std::uint32_t schema_version{1};
    std::string modulation{"QPSK"};
    double samples_per_symbol{4.0};
    std::optional<double> sample_rate_hz;
    std::optional<double> coarse_cfo_cycles_per_sample;
    std::optional<double> carrier_reference_cycles_per_sample;
    std::optional<double> phase_reference_radians;
    std::optional<double> noise_variance;
    std::string pulse_shape{"auto"};
    double rrc_rolloff{0.35};
    std::size_t acquisition_symbols{16};
    std::size_t maximum_buffered_samples{4U * 1024U * 1024U};
};

struct ReceiverResult {
    std::shared_ptr<std::vector<std::complex<float>>> symbols;
    std::shared_ptr<std::vector<std::uint8_t>> hard_bits;
    std::shared_ptr<std::vector<float>> soft_llrs;
    std::shared_ptr<std::vector<double>> sample_offsets;
    AcquisitionStatus acquisition_status{AcquisitionStatus::insufficient_data};
    core::EvidenceStatus mapping_status{core::EvidenceStatus::unverified};
    std::string modulation;
    std::size_t bits_per_symbol{0};
    double carrier_offset{0.0};
    std::string carrier_offset_unit{"cycles/sample"};
    double timing_offset_samples{0.0};
    double timing_error{0.0};
    double carrier_error{0.0};
    double evm_percent{100.0};
    double noise_variance{0.0};
    std::vector<double> unresolved_phase_rotations;
    std::vector<double> unresolved_carrier_offsets;
    std::vector<core::Diagnostic> diagnostics;
};

class ReceiverSession final {
public:
    explicit ReceiverSession(ReceiverConfig config);
    std::size_t process(std::span<const std::complex<float>> samples);
    [[nodiscard]] ReceiverResult flush();
    void reset();
    [[nodiscard]] std::size_t buffered_samples() const noexcept;

private:
    ReceiverConfig config_;
    std::vector<std::complex<float>> samples_;
    bool flushed_{false};
};

[[nodiscard]] ReceiverResult demodulate(
    std::span<const std::complex<float>> samples,
    const ReceiverConfig& config);

}  // namespace sih::receiver
