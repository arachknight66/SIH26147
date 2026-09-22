#include "sih/dsp/spectral.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace sih::dsp {
namespace {

[[nodiscard]] bool is_power_of_two(const std::size_t value) {
    return value > 0U && (value & (value - 1U)) == 0U;
}

void fft_in_place(std::vector<std::complex<double>>& values) {
    const auto size = values.size();
    for (std::size_t index = 1, reversed = 0; index < size; ++index) {
        std::size_t bit = size >> 1U;
        while ((reversed & bit) != 0U) {
            reversed ^= bit;
            bit >>= 1U;
        }
        reversed ^= bit;
        if (index < reversed) {
            std::swap(values[index], values[reversed]);
        }
    }
    for (std::size_t length = 2; length <= size; length <<= 1U) {
        const auto angle = -2.0 * std::numbers::pi / static_cast<double>(length);
        const std::complex<double> root{std::cos(angle), std::sin(angle)};
        for (std::size_t offset = 0; offset < size; offset += length) {
            std::complex<double> rotation{1.0, 0.0};
            for (std::size_t index = 0; index < length / 2U; ++index) {
                const auto even = values[offset + index];
                const auto odd = values[offset + index + length / 2U] * rotation;
                values[offset + index] = even + odd;
                values[offset + index + length / 2U] = even - odd;
                rotation *= root;
            }
        }
    }
}

[[nodiscard]] double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2U != 0U) {
        return *middle;
    }
    const auto lower = *std::max_element(values.begin(), middle);
    return (lower + *middle) / 2.0;
}

}  // namespace

SpectralAnalyzer::SpectralAnalyzer(SpectralConfig config) : config_(std::move(config)) {
    if (config_.schema_version != 1U) {
        throw std::invalid_argument("unsupported spectral configuration schema version");
    }
    if (!is_power_of_two(config_.fft_size) || config_.fft_size < 16U ||
        config_.fft_size > (1U << 20U)) {
        throw std::invalid_argument("FFT size must be a power of two in [16, 1048576]");
    }
    if (config_.hop_size == 0U || config_.hop_size > config_.fft_size) {
        throw std::invalid_argument("hop size must be in [1, fft_size]");
    }
    if (config_.sample_rate_hz &&
        (!std::isfinite(*config_.sample_rate_hz) || *config_.sample_rate_hz <= 0.0)) {
        throw std::invalid_argument("sample rate must be finite and positive when supplied");
    }
    const auto bins = config_.complex_input ? config_.fft_size : config_.fft_size / 2U + 1U;
    window_.resize(config_.fft_size);
    for (std::size_t index = 0; index < config_.fft_size; ++index) {
        window_[index] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(index) /
                                static_cast<double>(config_.fft_size - 1U)));
    }
    window_energy_ = std::inner_product(window_.begin(), window_.end(), window_.begin(), 0.0);
    psd_sum_.assign(bins, 0.0);
    frequencies_ = std::make_shared<std::vector<double>>(bins);
    times_ = std::make_shared<std::vector<double>>();
    stft_power_ = std::make_shared<std::vector<float>>();
    const double scale = config_.sample_rate_hz.value_or(1.0);
    if (config_.complex_input) {
        for (std::size_t bin = 0; bin < bins; ++bin) {
            const auto signed_bin = static_cast<std::int64_t>(bin) -
                                    static_cast<std::int64_t>(config_.fft_size / 2U);
            (*frequencies_)[bin] = static_cast<double>(signed_bin) * scale /
                                   static_cast<double>(config_.fft_size);
        }
    } else {
        for (std::size_t bin = 0; bin < bins; ++bin) {
            (*frequencies_)[bin] = static_cast<double>(bin) * scale /
                                   static_cast<double>(config_.fft_size);
        }
    }
}

void SpectralAnalyzer::process_frame(
    const std::uint64_t frame_start, const std::size_t valid_samples) {
    std::vector<std::complex<double>> spectrum(config_.fft_size, {0.0, 0.0});
    for (std::size_t index = 0; index < valid_samples; ++index) {
        spectrum[index] = static_cast<std::complex<double>>(pending_[index]) *
                          static_cast<double>(window_[index]);
    }
    fft_in_place(spectrum);
    const double frame_window_energy = valid_samples == config_.fft_size
        ? window_energy_
        : std::inner_product(
              window_.begin(), window_.begin() + static_cast<std::ptrdiff_t>(valid_samples),
              window_.begin(), 0.0);
    const double density_scale = 1.0 /
        (config_.sample_rate_hz.value_or(1.0) * std::max(frame_window_energy, 1.0e-30));
    const auto bins = psd_sum_.size();
    const bool retain_stft = times_->size() < config_.max_stft_frames;
    if (retain_stft) {
        const auto time_scale = config_.sample_rate_hz.value_or(1.0);
        times_->push_back((static_cast<double>(frame_start) +
                           static_cast<double>(valid_samples) / 2.0) / time_scale);
        stft_power_->reserve(stft_power_->size() + bins);
    } else {
        ++dropped_stft_frames_;
    }
    for (std::size_t output_bin = 0; output_bin < bins; ++output_bin) {
        std::size_t fft_bin = output_bin;
        if (config_.complex_input) {
            fft_bin = (output_bin + config_.fft_size / 2U) % config_.fft_size;
        }
        double power = std::norm(spectrum[fft_bin]) * density_scale;
        if (!config_.complex_input && output_bin != 0U &&
            output_bin != config_.fft_size / 2U) {
            power *= 2.0;
        }
        psd_sum_[output_bin] += power;
        if (retain_stft) {
            stft_power_->push_back(static_cast<float>(power));
        }
    }
    ++segment_count_;
}

void SpectralAnalyzer::update(const std::span<const std::complex<float>> input) {
    if (finished_) {
        throw std::logic_error("spectral analyzer has already been finished");
    }
    for (const auto sample : input) {
        if (!std::isfinite(sample.real()) || !std::isfinite(sample.imag())) {
            throw std::invalid_argument("spectral input contains NaN or infinity");
        }
        pending_.push_back(config_.complex_input ? sample : std::complex<float>{sample.real(), 0.0F});
        ++samples_received_;
        if (pending_.size() == config_.fft_size) {
            process_frame(next_frame_start_, config_.fft_size);
            pending_.erase(
                pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(config_.hop_size));
            next_frame_start_ += config_.hop_size;
        }
    }
}

void SpectralAnalyzer::finish(const bool include_partial) {
    if (!finished_ && include_partial && !pending_.empty() &&
        (segment_count_ == 0U || pending_.size() > config_.fft_size - config_.hop_size)) {
        process_frame(next_frame_start_, pending_.size());
    }
    finished_ = true;
}

SpectralResult SpectralAnalyzer::result() const {
    SpectralResult value;
    value.frequencies = std::make_shared<std::vector<double>>(*frequencies_);
    value.times = std::make_shared<std::vector<double>>(*times_);
    value.stft_power = std::make_shared<std::vector<float>>(*stft_power_);
    value.frequency_bins = psd_sum_.size();
    value.stft_frames = times_->size();
    value.processed_samples = samples_received_;
    value.welch_segments = segment_count_;
    value.dropped_stft_frames = dropped_stft_frames_;
    value.frequency_unit = config_.sample_rate_hz ? "Hz" : "cycles/sample";
    value.time_unit = config_.sample_rate_hz ? "s" : "samples";
    value.power_unit = config_.sample_rate_hz ? "power/Hz" : "power/(cycles/sample)";
    value.psd = std::make_shared<std::vector<float>>(psd_sum_.size(), 0.0F);
    if (segment_count_ > 0U) {
        for (std::size_t bin = 0; bin < psd_sum_.size(); ++bin) {
            (*value.psd)[bin] = static_cast<float>(
                psd_sum_[bin] / static_cast<double>(segment_count_));
        }
    }
    return value;
}

void SpectralAnalyzer::reset() {
    pending_.clear();
    std::fill(psd_sum_.begin(), psd_sum_.end(), 0.0);
    times_->clear();
    stft_power_->clear();
    samples_received_ = 0;
    next_frame_start_ = 0;
    segment_count_ = 0;
    dropped_stft_frames_ = 0;
    finished_ = false;
}

std::vector<SpectralBand> detect_spectral_bands(
    const std::span<const double> frequencies,
    const std::span<const float> psd,
    const double threshold_above_noise_db,
    const std::size_t minimum_bins) {
    if (frequencies.size() != psd.size() || frequencies.size() < 2U ||
        minimum_bins == 0U || !std::isfinite(threshold_above_noise_db)) {
        throw std::invalid_argument("band detection inputs/configuration are invalid");
    }
    std::vector<double> finite_power;
    finite_power.reserve(psd.size());
    for (const auto value : psd) {
        if (!std::isfinite(value) || value < 0.0F) {
            throw std::invalid_argument("PSD values must be finite and nonnegative");
        }
        finite_power.push_back(static_cast<double>(value));
    }
    const double noise = std::max(median(finite_power), std::numeric_limits<double>::min());
    const double threshold = noise * std::pow(10.0, threshold_above_noise_db / 10.0);
    const double bin_width = std::abs(frequencies[1] - frequencies[0]);
    std::vector<SpectralBand> result;
    std::size_t begin = 0;
    while (begin < psd.size()) {
        while (begin < psd.size() && psd[begin] < threshold) {
            ++begin;
        }
        auto end = begin;
        while (end < psd.size() && psd[end] >= threshold) {
            ++end;
        }
        if (end - begin >= minimum_bins) {
            double total = 0.0;
            double weighted = 0.0;
            double peak = 0.0;
            for (auto index = begin; index < end; ++index) {
                total += psd[index];
                weighted += frequencies[index] * psd[index];
                peak = std::max(peak, static_cast<double>(psd[index]));
            }
            double cumulative = 0.0;
            auto low = begin;
            auto high = end - 1U;
            while (low < end && cumulative + psd[low] < total * 0.005) {
                cumulative += psd[low++];
            }
            cumulative = 0.0;
            while (high > begin && cumulative + psd[high] < total * 0.005) {
                cumulative += psd[high--];
            }
            result.push_back(SpectralBand{
                frequencies[begin] - bin_width / 2.0,
                frequencies[end - 1U] + bin_width / 2.0,
                total > 0.0 ? weighted / total : 0.0,
                frequencies[high] - frequencies[low] + bin_width,
                total * bin_width,
                10.0 * std::log10(std::max(peak / noise, 1.0))});
        }
        begin = end + (end == begin ? 1U : 0U);
    }
    return result;
}

}  // namespace sih::dsp
