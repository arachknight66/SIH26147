#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <vector>

namespace sih::dsp {

[[nodiscard]] std::vector<float> design_lowpass_fir(
    double cutoff_cycles_per_sample, std::size_t tap_count);

[[nodiscard]] std::vector<float> design_bandpass_fir(
    double lower_cycles_per_sample,
    double upper_cycles_per_sample,
    std::size_t tap_count);

struct PreprocessingConfig {
    std::uint32_t schema_version{1};
    bool remove_dc{false};
    float dc_pole{0.995F};
    bool enable_agc{false};
    float agc_target_rms{0.5F};
    float agc_smoothing{0.001F};
    double mix_frequency_cycles_per_sample{0.0};
    std::vector<float> fir_taps;
    std::uint32_t resample_up{1};
    std::uint32_t resample_down{1};
};

struct PreprocessingChunk {
    std::shared_ptr<std::vector<std::complex<float>>> samples{
        std::make_shared<std::vector<std::complex<float>>>()};
    std::uint64_t input_samples_consumed{0};
    std::uint64_t output_samples_produced{0};
};

class PreprocessorSession final {
public:
    explicit PreprocessorSession(PreprocessingConfig config);

    [[nodiscard]] PreprocessingChunk process(std::span<const std::complex<float>> input);
    [[nodiscard]] PreprocessingChunk finish();
    void reset() noexcept;

private:
    [[nodiscard]] std::complex<float> preprocess_one(std::complex<float> sample);
    void append_resampled(std::complex<float> sample, std::vector<std::complex<float>>& output);
    void emit_until(std::uint64_t maximum_time, std::vector<std::complex<float>>& output);
    [[nodiscard]] std::complex<float> resample_at(std::uint64_t time) const;
    void trim_resample_history();

    PreprocessingConfig config_;
    std::vector<float> resample_taps_;
    std::complex<float> previous_input_{0.0F, 0.0F};
    std::complex<float> previous_dc_output_{0.0F, 0.0F};
    double mixer_phase_{0.0};
    double agc_power_{1.0};
    std::deque<std::complex<float>> fir_history_;
    std::deque<std::complex<float>> resample_history_;
    std::uint64_t resample_history_start_{0};
    std::uint64_t resample_input_count_{0};
    std::uint64_t next_output_index_{0};
    std::uint64_t input_count_{0};
    std::uint64_t output_count_{0};
    bool finished_{false};
};

struct TimeStatistics {
    std::uint64_t sample_count{0};
    double mean_i{0.0};
    double mean_q{0.0};
    double variance_i{0.0};
    double variance_q{0.0};
    double rms_amplitude{0.0};
    double peak_amplitude{0.0};
    double crest_factor{0.0};
    bool clipping_available{true};
    double clipping_fraction{0.0};
};

class StatisticsAccumulator final {
public:
    explicit StatisticsAccumulator(float clipping_level = 0.999F);
    void update(std::span<const std::complex<float>> input);
    [[nodiscard]] TimeStatistics result() const noexcept;
    void reset() noexcept;

private:
    float clipping_level_;
    bool clipping_available_{true};
    std::uint64_t count_{0};
    std::uint64_t clipped_{0};
    long double sum_i_{0.0};
    long double sum_q_{0.0};
    long double sum_i2_{0.0};
    long double sum_q2_{0.0};
    long double sum_magnitude2_{0.0};
    double peak_{0.0};
};

}  // namespace sih::dsp
