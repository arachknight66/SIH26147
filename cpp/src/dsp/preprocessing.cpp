#include "sih/dsp/preprocessing.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace sih::dsp {
namespace {

[[nodiscard]] double ideal_lowpass(const double cutoff, const double position) {
    return position == 0.0
        ? 2.0 * cutoff
        : std::sin(2.0 * std::numbers::pi * cutoff * position) /
              (std::numbers::pi * position);
}

[[nodiscard]] std::vector<float> make_resample_filter(
    const std::uint32_t up, const std::uint32_t down) {
    if (up == 0U || down == 0U || up > 1024U || down > 1024U) {
        throw std::invalid_argument("resampling factors must be in [1, 1024]");
    }
    if (up == 1U && down == 1U) {
        return {1.0F};
    }
    const auto scale = std::max(up, down);
    const std::size_t half = 16U * scale;
    const std::size_t length = 2U * half + 1U;
    const double cutoff = 0.5 / static_cast<double>(scale);
    std::vector<float> taps(length);
    for (std::size_t index = 0; index < length; ++index) {
        const double x = static_cast<double>(index) - static_cast<double>(half);
        const double sinc = x == 0.0
            ? 2.0 * cutoff
            : std::sin(2.0 * std::numbers::pi * cutoff * x) / (std::numbers::pi * x);
        const double window = 0.5 - 0.5 * std::cos(
            2.0 * std::numbers::pi * static_cast<double>(index) /
            static_cast<double>(length - 1U));
        taps[index] = static_cast<float>(sinc * window * static_cast<double>(up));
    }
    return taps;
}

}  // namespace

std::vector<float> design_lowpass_fir(
    const double cutoff_cycles_per_sample, const std::size_t tap_count) {
    if (!(cutoff_cycles_per_sample > 0.0 && cutoff_cycles_per_sample < 0.5) ||
        tap_count < 3U || tap_count % 2U == 0U) {
        throw std::invalid_argument("low-pass cutoff must be in (0, 0.5) and tap count must be odd and >= 3");
    }
    std::vector<float> taps(tap_count);
    const double center = static_cast<double>(tap_count - 1U) / 2.0;
    for (std::size_t index = 0; index < tap_count; ++index) {
        const double position = static_cast<double>(index) - center;
        const double window = 0.54 - 0.46 * std::cos(
            2.0 * std::numbers::pi * static_cast<double>(index) /
            static_cast<double>(tap_count - 1U));
        taps[index] = static_cast<float>(ideal_lowpass(cutoff_cycles_per_sample, position) * window);
    }
    const double gain = std::accumulate(taps.begin(), taps.end(), 0.0);
    for (auto& tap : taps) {
        tap = static_cast<float>(tap / gain);
    }
    return taps;
}

std::vector<float> design_bandpass_fir(
    const double lower_cycles_per_sample,
    const double upper_cycles_per_sample,
    const std::size_t tap_count) {
    if (!(lower_cycles_per_sample >= 0.0 &&
          lower_cycles_per_sample < upper_cycles_per_sample &&
          upper_cycles_per_sample < 0.5) || tap_count < 3U || tap_count % 2U == 0U) {
        throw std::invalid_argument("band-pass edges must satisfy 0 <= lower < upper < 0.5 and tap count must be odd and >= 3");
    }
    std::vector<float> taps(tap_count);
    const double center = static_cast<double>(tap_count - 1U) / 2.0;
    for (std::size_t index = 0; index < tap_count; ++index) {
        const double position = static_cast<double>(index) - center;
        const double window = 0.54 - 0.46 * std::cos(
            2.0 * std::numbers::pi * static_cast<double>(index) /
            static_cast<double>(tap_count - 1U));
        taps[index] = static_cast<float>(
            (ideal_lowpass(upper_cycles_per_sample, position) -
             ideal_lowpass(lower_cycles_per_sample, position)) * window);
    }
    return taps;
}

PreprocessorSession::PreprocessorSession(PreprocessingConfig config)
    : config_(std::move(config)),
      resample_taps_(make_resample_filter(config_.resample_up, config_.resample_down)) {
    if (config_.schema_version != 1U) {
        throw std::invalid_argument("unsupported preprocessing configuration schema version");
    }
    if (!(config_.dc_pole >= 0.0F && config_.dc_pole < 1.0F)) {
        throw std::invalid_argument("DC blocker pole must be in [0, 1)");
    }
    if (!(config_.agc_target_rms > 0.0F) ||
        !(config_.agc_smoothing > 0.0F && config_.agc_smoothing <= 1.0F)) {
        throw std::invalid_argument("AGC target must be positive and smoothing must be in (0, 1]");
    }
    if (!std::isfinite(config_.mix_frequency_cycles_per_sample) ||
        std::abs(config_.mix_frequency_cycles_per_sample) > 0.5) {
        throw std::invalid_argument("mix frequency must be finite and within [-0.5, 0.5] cycles/sample");
    }
    if (config_.fir_taps.empty()) {
        config_.fir_taps = {1.0F};
    }
    if (!std::all_of(config_.fir_taps.begin(), config_.fir_taps.end(),
                     [](const float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("FIR taps must be finite");
    }
}

std::complex<float> PreprocessorSession::preprocess_one(std::complex<float> sample) {
    if (config_.remove_dc) {
        const auto output = sample - previous_input_ + config_.dc_pole * previous_dc_output_;
        previous_input_ = sample;
        previous_dc_output_ = output;
        sample = output;
    }
    if (config_.mix_frequency_cycles_per_sample != 0.0) {
        const auto oscillator = std::polar(
            1.0F, static_cast<float>(-2.0 * std::numbers::pi * mixer_phase_));
        sample *= oscillator;
        mixer_phase_ += config_.mix_frequency_cycles_per_sample;
        mixer_phase_ -= std::floor(mixer_phase_);
    }
    if (config_.enable_agc) {
        const double power = static_cast<double>(std::norm(sample));
        agc_power_ += static_cast<double>(config_.agc_smoothing) * (power - agc_power_);
        const double gain = static_cast<double>(config_.agc_target_rms) /
                            std::sqrt(std::max(agc_power_, 1.0e-20));
        sample *= static_cast<float>(gain);
    }
    fir_history_.push_front(sample);
    while (fir_history_.size() > config_.fir_taps.size()) {
        fir_history_.pop_back();
    }
    std::complex<float> filtered{0.0F, 0.0F};
    for (std::size_t tap = 0; tap < fir_history_.size(); ++tap) {
        filtered += config_.fir_taps[tap] * fir_history_[tap];
    }
    return filtered;
}

std::complex<float> PreprocessorSession::resample_at(const std::uint64_t time) const {
    std::complex<float> value{0.0F, 0.0F};
    for (std::size_t tap = 0; tap < resample_taps_.size(); ++tap) {
        if (time < tap) {
            continue;
        }
        const auto delta = time - tap;
        if (delta % config_.resample_up != 0U) {
            continue;
        }
        const auto source_index = delta / config_.resample_up;
        if (source_index < resample_history_start_ || source_index >= resample_input_count_) {
            continue;
        }
        value += resample_taps_[tap] *
                 resample_history_[static_cast<std::size_t>(source_index - resample_history_start_)];
    }
    return value;
}

void PreprocessorSession::trim_resample_history() {
    const auto next_time = next_output_index_ * config_.resample_down;
    const auto earliest = next_time >= resample_taps_.size() - 1U
        ? (next_time - (resample_taps_.size() - 1U)) / config_.resample_up
        : 0U;
    while (!resample_history_.empty() && resample_history_start_ < earliest) {
        resample_history_.pop_front();
        ++resample_history_start_;
    }
}

void PreprocessorSession::emit_until(
    const std::uint64_t maximum_time, std::vector<std::complex<float>>& output) {
    while (next_output_index_ * config_.resample_down <= maximum_time) {
        output.push_back(resample_at(next_output_index_ * config_.resample_down));
        ++next_output_index_;
        ++output_count_;
        trim_resample_history();
    }
}

void PreprocessorSession::append_resampled(
    const std::complex<float> sample, std::vector<std::complex<float>>& output) {
    resample_history_.push_back(sample);
    ++resample_input_count_;
    emit_until((resample_input_count_ - 1U) * config_.resample_up, output);
}

PreprocessingChunk PreprocessorSession::process(const std::span<const std::complex<float>> input) {
    if (finished_) {
        throw std::logic_error("preprocessor session has already been finished");
    }
    PreprocessingChunk result;
    result.samples->reserve(
        static_cast<std::size_t>(std::ceil(
            static_cast<double>(input.size()) * config_.resample_up / config_.resample_down)) + 2U);
    for (const auto sample : input) {
        if (!std::isfinite(sample.real()) || !std::isfinite(sample.imag())) {
            throw std::invalid_argument("preprocessing input contains NaN or infinity");
        }
        append_resampled(preprocess_one(sample), *result.samples);
        ++input_count_;
    }
    result.input_samples_consumed = input_count_;
    result.output_samples_produced = output_count_;
    return result;
}

PreprocessingChunk PreprocessorSession::finish() {
    PreprocessingChunk result;
    if (!finished_ && resample_input_count_ > 0U) {
        const auto final_time = (resample_input_count_ - 1U) * config_.resample_up +
                                (resample_taps_.size() - 1U);
        emit_until(final_time, *result.samples);
    }
    finished_ = true;
    result.input_samples_consumed = input_count_;
    result.output_samples_produced = output_count_;
    return result;
}

void PreprocessorSession::reset() noexcept {
    previous_input_ = {0.0F, 0.0F};
    previous_dc_output_ = {0.0F, 0.0F};
    mixer_phase_ = 0.0;
    agc_power_ = 1.0;
    fir_history_.clear();
    resample_history_.clear();
    resample_history_start_ = 0;
    resample_input_count_ = 0;
    next_output_index_ = 0;
    input_count_ = 0;
    output_count_ = 0;
    finished_ = false;
}

StatisticsAccumulator::StatisticsAccumulator(const float clipping_level)
    : clipping_level_(clipping_level), clipping_available_(std::isfinite(clipping_level)) {
    if (!(clipping_level_ > 0.0F)) {
        throw std::invalid_argument("clipping level must be positive");
    }
}

void StatisticsAccumulator::update(const std::span<const std::complex<float>> input) {
    for (const auto sample : input) {
        if (!std::isfinite(sample.real()) || !std::isfinite(sample.imag())) {
            throw std::invalid_argument("statistics input contains NaN or infinity");
        }
        const auto i = static_cast<long double>(sample.real());
        const auto q = static_cast<long double>(sample.imag());
        sum_i_ += i;
        sum_q_ += q;
        sum_i2_ += i * i;
        sum_q2_ += q * q;
        const auto magnitude2 = i * i + q * q;
        sum_magnitude2_ += magnitude2;
        peak_ = std::max(peak_, std::sqrt(static_cast<double>(magnitude2)));
        if (clipping_available_ && (std::abs(sample.real()) >= clipping_level_ ||
            std::abs(sample.imag()) >= clipping_level_)) {
            ++clipped_;
        }
        ++count_;
    }
}

TimeStatistics StatisticsAccumulator::result() const noexcept {
    TimeStatistics value;
    value.sample_count = count_;
    value.clipping_available = clipping_available_;
    if (count_ == 0U) {
        return value;
    }
    const auto count = static_cast<long double>(count_);
    value.mean_i = static_cast<double>(sum_i_ / count);
    value.mean_q = static_cast<double>(sum_q_ / count);
    value.variance_i = std::max(0.0, static_cast<double>(sum_i2_ / count) - value.mean_i * value.mean_i);
    value.variance_q = std::max(0.0, static_cast<double>(sum_q2_ / count) - value.mean_q * value.mean_q);
    value.rms_amplitude = std::sqrt(static_cast<double>(sum_magnitude2_ / count));
    value.peak_amplitude = peak_;
    value.crest_factor = value.rms_amplitude > 0.0 ? peak_ / value.rms_amplitude : 0.0;
    value.clipping_fraction = clipping_available_
        ? static_cast<double>(clipped_) / static_cast<double>(count_)
        : 0.0;
    return value;
}

void StatisticsAccumulator::reset() noexcept {
    count_ = 0;
    clipped_ = 0;
    sum_i_ = 0.0;
    sum_q_ = 0.0;
    sum_i2_ = 0.0;
    sum_q2_ = 0.0;
    sum_magnitude2_ = 0.0;
    peak_ = 0.0;
}

}  // namespace sih::dsp
