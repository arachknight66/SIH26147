#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sih::dsp {

struct SpectralConfig {
    std::uint32_t schema_version{1};
    std::size_t fft_size{1024};
    std::size_t hop_size{512};
    bool complex_input{true};
    std::optional<double> sample_rate_hz;
    std::size_t max_stft_frames{512};
};

struct SpectralResult {
    std::shared_ptr<std::vector<double>> frequencies;
    std::shared_ptr<std::vector<float>> psd;
    std::shared_ptr<std::vector<double>> times;
    std::shared_ptr<std::vector<float>> stft_power;
    std::size_t frequency_bins{0};
    std::size_t stft_frames{0};
    std::uint64_t processed_samples{0};
    std::uint64_t welch_segments{0};
    std::uint64_t dropped_stft_frames{0};
    std::string frequency_unit;
    std::string time_unit;
    std::string power_unit;
};

struct SpectralBand {
    double lower_frequency{0.0};
    double upper_frequency{0.0};
    double center_frequency{0.0};
    double occupied_bandwidth{0.0};
    double integrated_power{0.0};
    double peak_to_noise_db{0.0};
};

class SpectralAnalyzer final {
public:
    explicit SpectralAnalyzer(SpectralConfig config);

    void update(std::span<const std::complex<float>> input);
    void finish(bool include_partial = true);
    [[nodiscard]] SpectralResult result() const;
    void reset();

private:
    void process_frame(std::uint64_t frame_start, std::size_t valid_samples);

    SpectralConfig config_;
    std::vector<float> window_;
    double window_energy_{0.0};
    std::vector<std::complex<float>> pending_;
    std::vector<double> psd_sum_;
    std::shared_ptr<std::vector<double>> frequencies_;
    std::shared_ptr<std::vector<double>> times_;
    std::shared_ptr<std::vector<float>> stft_power_;
    std::uint64_t samples_received_{0};
    std::uint64_t next_frame_start_{0};
    std::uint64_t segment_count_{0};
    std::uint64_t dropped_stft_frames_{0};
    bool finished_{false};
};

[[nodiscard]] std::vector<SpectralBand> detect_spectral_bands(
    std::span<const double> frequencies,
    std::span<const float> psd,
    double threshold_above_noise_db = 8.0,
    std::size_t minimum_bins = 2);

}  // namespace sih::dsp
