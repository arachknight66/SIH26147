#include "sih/estimation/analysis.hpp"

#include "sih/dsp/spectral.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sih::estimation {
namespace {

constexpr double pi = 3.14159265358979323846;

double clamp01(const double value) { return std::clamp(value, 0.0, 1.0); }

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
    std::nth_element(values.begin(), middle, values.end());
    double result = *middle;
    if (values.size() % 2U == 0U) {
        const auto lower = std::max_element(values.begin(), middle);
        result = (result + *lower) / 2.0;
    }
    return result;
}

std::vector<std::complex<double>> normalize(std::span<const std::complex<float>> input) {
    std::complex<double> mean{0.0, 0.0};
    for (const auto value : input) mean += static_cast<std::complex<double>>(value);
    mean /= static_cast<double>(input.size());
    double power = 0.0;
    for (const auto value : input) power += std::norm(static_cast<std::complex<double>>(value) - mean);
    power /= static_cast<double>(input.size());
    const double scale = std::sqrt(power);
    std::vector<std::complex<double>> output;
    output.reserve(input.size());
    for (const auto value : input) output.push_back((static_cast<std::complex<double>>(value) - mean) / scale);
    return output;
}

double power_cfo(const std::vector<std::complex<double>>& samples, const unsigned order) {
    std::complex<double> correlation{0.0, 0.0};
    auto previous = std::pow(samples.front(), order);
    for (std::size_t index = 1; index < samples.size(); ++index) {
        const auto current = std::pow(samples[index], order);
        correlation += current * std::conj(previous);
        previous = current;
    }
    const double coarse = std::arg(correlation) / (2.0 * pi * static_cast<double>(order));
    std::vector<double> estimates{coarse};
    for (const std::size_t lag : {16U, 64U, 256U}) {
        if (samples.size() <= lag * 4U) continue;
        std::complex<double> delayed{0.0, 0.0};
        for (std::size_t index = 0; index + lag < samples.size(); ++index) {
            delayed += std::pow(samples[index + lag], order) *
                       std::conj(std::pow(samples[index], order));
        }
        const double wrapped = std::arg(delayed) /
            (2.0 * pi * static_cast<double>(order) * static_cast<double>(lag));
        const double spacing = 1.0 / (static_cast<double>(order) * static_cast<double>(lag));
        estimates.push_back(wrapped + std::round((coarse - wrapped) / spacing) * spacing);
    }
    return median(estimates);
}

std::vector<double> transition_energy(
    const std::vector<std::complex<double>>& samples, const double cfo) {
    std::vector<double> result(samples.size(), 0.0);
    const auto rotation = std::polar(1.0, -2.0 * pi * cfo);
    for (std::size_t index = 1; index < samples.size(); ++index) {
        result[index] = std::norm(samples[index] * rotation - samples[index - 1]);
    }
    const auto center = median(result);
    for (auto& value : result) value -= center;
    return result;
}

double periodic_score(const std::vector<double>& metric, const double period) {
    std::complex<double> total{0.0, 0.0};
    double magnitude = 0.0;
    for (std::size_t index = 0; index < metric.size(); ++index) {
        total += metric[index] * std::polar(1.0, -2.0 * pi * static_cast<double>(index) / period);
        magnitude += std::abs(metric[index]);
    }
    return magnitude > 0.0 ? std::abs(total) / magnitude : 0.0;
}

std::vector<RateCandidate> estimate_rates(
    const std::vector<std::complex<double>>& samples, const AnalysisConfig& config, const double cfo) {
    const auto metric = transition_energy(samples, cfo);
    std::vector<double> frequency_transition(samples.size(), 0.0);
    std::vector<double> phase_increment(samples.size(), 0.0);
    for (std::size_t index = 1; index < samples.size(); ++index) {
        phase_increment[index] = std::arg(samples[index] * std::conj(samples[index - 1]));
    }
    for (std::size_t index = 2; index < samples.size(); ++index) {
        const double delta = std::remainder(phase_increment[index] - phase_increment[index - 1], 2.0 * pi);
        frequency_transition[index] = delta * delta;
    }
    const auto frequency_center = median(frequency_transition);
    for (auto& value : frequency_transition) value -= frequency_center;
    std::vector<RateCandidate> candidates;
    const auto first = static_cast<int>(std::ceil(config.minimum_samples_per_symbol * 20.0));
    const auto last = static_cast<int>(std::floor(config.maximum_samples_per_symbol * 20.0));
    for (int step = first; step <= last; ++step) {
        const double sps = static_cast<double>(step) / 20.0;
        candidates.push_back({1.0 / sps, sps,
            std::max(periodic_score(metric, sps), periodic_score(frequency_transition, sps))});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.score > right.score;
    });
    const double peak = candidates.empty() ? 0.0 : candidates.front().score;
    std::vector<RateCandidate> distinct;
    if (!candidates.empty()) {
        auto fundamental = candidates.front();
        for (const auto& candidate : candidates) {
            if (candidate.score >= peak * 0.97 && candidate.samples_per_symbol > fundamental.samples_per_symbol) {
                fundamental = candidate;
            }
        }
        distinct.push_back(fundamental);
    }
    for (const auto& candidate : candidates) {
        if (std::none_of(distinct.begin(), distinct.end(), [&](const auto& present) {
                return std::abs(present.samples_per_symbol - candidate.samples_per_symbol) < 0.35;
            })) {
            distinct.push_back(candidate);
            if (distinct.size() == 5U) break;
        }
    }
    return distinct;
}

double phase_moment(const std::vector<std::complex<double>>& samples, const unsigned order) {
    std::complex<double> total{0.0, 0.0};
    std::size_t used = 0;
    for (const auto value : samples) {
        const auto magnitude = std::abs(value);
        if (magnitude > 0.15) {
            total += std::pow(value / magnitude, order);
            ++used;
        }
    }
    return used ? std::abs(total) / static_cast<double>(used) : 0.0;
}

std::vector<std::complex<double>> symbol_samples(
    const std::vector<std::complex<double>>& samples, const double sps, const double cfo,
    const double phase) {
    // A frequency estimate that predicts fewer than four cycles of rotation
    // across this observation is below the reliable resolution of this
    // preview. Treating it as exact can smear an otherwise stationary
    // constellation across several phase states.
    const double applied_cfo = std::abs(cfo) * static_cast<double>(samples.size()) < 4.0 ? 0.0 : cfo;
    std::vector<std::complex<double>> output;
    for (double index = phase; index + 1.0 < static_cast<double>(samples.size()); index += sps) {
        const auto left = static_cast<std::size_t>(index);
        const double fraction = index - static_cast<double>(left);
        const auto interpolated = samples[left] * (1.0 - fraction) + samples[left + 1U] * fraction;
        output.push_back(interpolated * std::polar(1.0, -2.0 * pi * applied_cfo * index));
    }
    return output;
}

std::vector<std::complex<double>> psk_points(const unsigned order) {
    std::vector<std::complex<double>> points;
    for (unsigned index = 0; index < order; ++index) {
        points.push_back(std::polar(1.0, 2.0 * pi * static_cast<double>(index) / order));
    }
    return points;
}

std::vector<std::complex<double>> qam_points(const unsigned side) {
    std::vector<std::complex<double>> points;
    double power = 0.0;
    for (unsigned row = 0; row < side; ++row) {
        for (unsigned column = 0; column < side; ++column) {
            const std::complex<double> value{
                2.0 * static_cast<double>(column) - static_cast<double>(side) + 1.0,
                2.0 * static_cast<double>(row) - static_cast<double>(side) + 1.0};
            points.push_back(value);
            power += std::norm(value);
        }
    }
    const double rms = std::sqrt(power / static_cast<double>(points.size()));
    for (auto& point : points) point /= rms;
    return points;
}

struct FitResult { double error{10.0}; double occupancy{0.0}; };

FitResult fit_constellation(
    const std::vector<std::complex<double>>& symbols,
    const std::vector<std::complex<double>>& points, const unsigned rotational_order) {
    if (symbols.size() < 32U) return {};
    double symbol_power = 0.0;
    for (const auto value : symbols) symbol_power += std::norm(value);
    const double scale = std::sqrt(symbol_power / static_cast<double>(symbols.size()));
    FitResult best;
    const std::size_t sample_step = std::max<std::size_t>(1U, symbols.size() / 1024U);
    const std::size_t used_symbols = (symbols.size() + sample_step - 1U) / sample_step;
    for (unsigned step = 0; step < 24U; ++step) {
        const double angle = (2.0 * pi / static_cast<double>(rotational_order)) *
                             static_cast<double>(step) / 24.0;
        const auto rotation = std::polar(1.0, -angle);
        double error = 0.0;
        std::vector<std::size_t> counts(points.size(), 0U);
        for (std::size_t symbol_index = 0; symbol_index < symbols.size(); symbol_index += sample_step) {
            const auto raw = symbols[symbol_index];
            const auto value = raw * rotation / scale;
            double nearest = std::numeric_limits<double>::infinity();
            std::size_t nearest_index = 0U;
            for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
                const double distance = std::norm(value - points[point_index]);
                if (distance < nearest) { nearest = distance; nearest_index = point_index; }
            }
            ++counts[nearest_index];
            error += std::min(nearest, 4.0);
        }
        double entropy = 0.0;
        for (const auto count : counts) if (count) {
            const double probability = static_cast<double>(count) / static_cast<double>(used_symbols);
            entropy -= probability * std::log(probability);
        }
        const double occupancy = points.size() > 1U ? entropy / std::log(static_cast<double>(points.size())) : 1.0;
        const double normalized_error = error / static_cast<double>(used_symbols);
        // A single phase sample can fit one occupied constellation point
        // extremely well while carrying no modulation-order evidence. Prefer
        // similarly good timing/phase fits that show the expected occupancy.
        const double selection_cost = normalized_error + 0.02 * (1.0 - occupancy);
        const double best_cost = best.error + 0.02 * (1.0 - best.occupancy);
        if (selection_cost < best_cost) best = {normalized_error, occupancy};
    }
    return best;
}

struct KMeansResult {
    double variance{1.0};
    double separation{0.0};
    double minimum_occupancy{0.0};
    double spacing_uniformity{0.0};
    double persistence{0.0};
};

KMeansResult cluster_frequency(const std::vector<std::complex<double>>& samples, const unsigned states) {
    std::vector<double> frequency;
    frequency.reserve(samples.size() - 1U);
    for (std::size_t index = 1; index < samples.size(); ++index) {
        frequency.push_back(std::arg(samples[index] * std::conj(samples[index - 1])) / (2.0 * pi));
    }
    const auto frequency_in_time = frequency;
    std::sort(frequency.begin(), frequency.end());
    std::vector<double> centers(states);
    for (unsigned state = 0; state < states; ++state) {
        centers[state] = frequency[(2U * state + 1U) * frequency.size() / (2U * states)];
    }
    std::vector<unsigned> assignments(frequency.size());
    std::vector<std::size_t> counts(states);
    for (unsigned iteration = 0; iteration < 20U; ++iteration) {
        std::fill(counts.begin(), counts.end(), 0U);
        std::vector<double> sums(states, 0.0);
        for (std::size_t index = 0; index < frequency.size(); ++index) {
            unsigned best = 0;
            for (unsigned state = 1; state < states; ++state) {
                if (std::abs(frequency[index] - centers[state]) < std::abs(frequency[index] - centers[best])) best = state;
            }
            assignments[index] = best; ++counts[best]; sums[best] += frequency[index];
        }
        for (unsigned state = 0; state < states; ++state) if (counts[state]) centers[state] = sums[state] / counts[state];
    }
    double residual = 0.0;
    for (std::size_t index = 0; index < frequency.size(); ++index) residual += std::pow(frequency[index] - centers[assignments[index]], 2);
    std::sort(centers.begin(), centers.end());
    double separation = std::numeric_limits<double>::infinity();
    double maximum_separation = 0.0;
    for (unsigned state = 1; state < states; ++state) {
        const double gap = centers[state] - centers[state - 1U];
        separation = std::min(separation, gap);
        maximum_separation = std::max(maximum_separation, gap);
    }
    const auto minimum_count = *std::min_element(counts.begin(), counts.end());
    std::size_t persistent = 0U;
    unsigned previous_state = states;
    for (const auto value : frequency_in_time) {
        unsigned state = 0U;
        for (unsigned candidate = 1U; candidate < states; ++candidate) {
            if (std::abs(value - centers[candidate]) < std::abs(value - centers[state])) state = candidate;
        }
        if (state == previous_state) ++persistent;
        previous_state = state;
    }
    return {residual / static_cast<double>(frequency.size()), separation,
            static_cast<double>(minimum_count) / static_cast<double>(frequency.size()),
            maximum_separation > 0.0 ? separation / maximum_separation : 0.0,
            frequency_in_time.size() > 1U ? static_cast<double>(persistent) /
                static_cast<double>(frequency_in_time.size() - 1U) : 0.0};
}

std::vector<ModulationCandidate> classify(
    const std::vector<std::complex<double>>& samples, const double sps,
    const double generic_cfo, const double rate_score) {
    const auto fsk2 = cluster_frequency(samples, 2);
    const auto fsk4 = cluster_frequency(samples, 4);
    const auto fsk8 = cluster_frequency(samples, 8);
    const double raw_variance = [&] {
        std::vector<double> f;
        for (std::size_t i = 1; i < samples.size(); ++i) f.push_back(std::arg(samples[i] * std::conj(samples[i-1])) / (2*pi));
        const double center = std::accumulate(f.begin(), f.end(), 0.0) / f.size();
        double v=0; for(double x:f) v+=(x-center)*(x-center); return v/f.size();
    }();
    const double frequency_persistence = [&] {
        std::vector<double> f;
        for (std::size_t i = 1; i < samples.size(); ++i) f.push_back(std::arg(samples[i] * std::conj(samples[i-1])) / (2*pi));
        const double center = std::accumulate(f.begin(), f.end(), 0.0) / f.size();
        double covariance = 0.0;
        double variance = 0.0;
        for (std::size_t i = 1; i < f.size(); ++i) {
            covariance += (f[i] - center) * (f[i - 1U] - center);
            variance += (f[i] - center) * (f[i] - center);
        }
        return variance > 0.0 ? covariance / variance : 0.0;
    }();
    const double fsk2_gain = 1.0 - fsk2.variance / std::max(raw_variance, 1e-12);
    const double fsk4_gain = 1.0 - fsk4.variance / std::max(raw_variance, 1e-12);
    const double fsk2_score = clamp01((fsk2_gain - 0.65) / 0.25) * clamp01(fsk2.minimum_occupancy / 0.20) * clamp01(rate_score / 0.08) * clamp01((fsk2.persistence - 0.48) / 0.30);
    const double fsk4_raw = clamp01((fsk4_gain - 0.60) / 0.30) * clamp01(fsk4.minimum_occupancy / 0.10) * clamp01(rate_score / 0.08) * clamp01(fsk4.spacing_uniformity / 0.55) * clamp01((fsk4.persistence - 0.45) / 0.30);
    const double fsk8_gain = 1.0 - fsk8.variance / std::max(raw_variance, 1e-12);
    const double fsk8_raw = clamp01((fsk8_gain - 0.70) / 0.25) * clamp01(fsk8.minimum_occupancy / 0.045) * clamp01(rate_score / 0.08) * clamp01(fsk8.spacing_uniformity / 0.45) * clamp01((fsk8.persistence - 0.45) / 0.30);

    struct Model { const char* label; const char* family; unsigned order; unsigned qam_side; };
    constexpr std::array<Model, 6> models{{
        {"BPSK", "PSK", 2, 0}, {"QPSK", "PSK", 4, 0}, {"8PSK", "PSK", 8, 0},
        {"16-QAM", "QAM", 4, 4}, {"64-QAM", "QAM", 4, 8}, {"256-QAM", "QAM", 4, 16}}};
    std::vector<ModulationCandidate> result;
    for (const auto& model : models) {
        const double cfo = power_cfo(samples, model.order);
        FitResult best_fit;
        const auto phase_count = std::max<std::size_t>(2U, static_cast<std::size_t>(std::ceil(sps)));
        for (std::size_t phase = 0; phase < phase_count; ++phase) {
            const auto symbols = symbol_samples(samples, sps, cfo, static_cast<double>(phase));
            const auto points = model.qam_side ? qam_points(model.qam_side) : psk_points(model.order);
            const auto fit = fit_constellation(symbols, points, model.order);
            const double fit_cost = fit.error + 0.02 * (1.0 - fit.occupancy);
            const double best_cost = best_fit.error + 0.02 * (1.0 - best_fit.occupancy);
            if (fit_cost < best_cost) best_fit = fit;
        }
        double score = std::exp(-4.0 * best_fit.error) *
                       std::exp(-5.0 * (1.0 - best_fit.occupancy)) *
                       clamp01(rate_score / 0.06);
        (void)phase_moment;
        result.push_back({model.label, model.family, clamp01(score), InferenceStatus::candidate,
                          best_fit.error, 0.0,
                          {"CFO-tolerant constellation fit", "constellation occupancy", "transition-period rate evidence"}, {}});
    }
    unsigned selected_fsk_states = 8U;
    double selected_fsk_score = fsk8_raw;
    double selected_fsk_variance = fsk8.variance;
    if (fsk2_score >= 0.62) {
        selected_fsk_states = 2U; selected_fsk_score = fsk2_score; selected_fsk_variance = fsk2.variance;
    } else if (fsk4_raw >= 0.55) {
        selected_fsk_states = 4U; selected_fsk_score = fsk4_raw; selected_fsk_variance = fsk4.variance;
    }
    const double fsk_family_score = clamp01((frequency_persistence - 0.05) / 0.30) *
                                    clamp01(rate_score / 0.08);
    result.push_back({std::to_string(selected_fsk_states) + "-FSK", "FSK",
                      clamp01(std::max(1.35 * selected_fsk_score, fsk_family_score)), InferenceStatus::candidate,
                      selected_fsk_variance, 0.0,
                      {"instantaneous-frequency clustering", "tone-state dwell persistence", "lag-one frequency persistence"}, {}});
    (void)generic_cfo;
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) { return left.score > right.score; });
    return result;
}

double delayed_correlation_peak(const std::vector<std::complex<double>>& samples) {
    double best = 0.0;
    const auto maximum_lag = std::min<std::size_t>(512U, samples.size() / 4U);
    for (std::size_t lag = 16U; lag <= maximum_lag; ++lag) {
        std::complex<double> correlation{0.0, 0.0};
        double left_power = 0.0;
        double right_power = 0.0;
        for (std::size_t index = 0; index + lag < samples.size(); ++index) {
            correlation += samples[index + lag] * std::conj(samples[index]);
            left_power += std::norm(samples[index]);
            right_power += std::norm(samples[index + lag]);
        }
        best = std::max(best, std::abs(correlation) /
            std::sqrt(std::max(left_power * right_power, 1e-30)));
    }
    return best;
}

void spectral_estimates(
    std::span<const std::complex<float>> samples, const AnalysisConfig& config,
    Estimate& snr, Estimate& bandwidth) {
    dsp::SpectralConfig spectral;
    spectral.fft_size = 1024U;
    while (spectral.fft_size > samples.size() && spectral.fft_size > 64U) spectral.fft_size /= 2U;
    spectral.hop_size = spectral.fft_size / 2U;
    spectral.sample_rate_hz = config.sample_rate_hz;
    spectral.max_stft_frames = 0U;
    dsp::SpectralAnalyzer analyzer(spectral);
    analyzer.update(samples); analyzer.finish();
    const auto result = analyzer.result();
    std::vector<double> powers(result.psd->begin(), result.psd->end());
    const double floor = std::max(median(powers), 1e-20);
    std::vector<double> excess(powers.size());
    std::transform(powers.begin(), powers.end(), excess.begin(), [&](double p) { return std::max(0.0, p - floor); });
    const double signal_power = std::accumulate(excess.begin(), excess.end(), 0.0);
    if (signal_power <= floor * static_cast<double>(powers.size()) * 0.05) {
        snr = {std::nullopt, std::nullopt, "dB", Validity::unavailable, "no resolvable signal above PSD floor"};
        bandwidth = {std::nullopt, std::nullopt, result.frequency_unit, Validity::unavailable, "occupied power unavailable"};
        return;
    }
    std::vector<std::size_t> order(powers.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return (*result.frequencies)[a] < (*result.frequencies)[b]; });
    const double tail = signal_power * 0.005;
    double cumulative = 0.0; std::size_t low = 0;
    while (low + 1U < order.size() && cumulative + excess[order[low]] < tail) cumulative += excess[order[low++]];
    cumulative = 0.0; std::size_t high = order.size() - 1U;
    while (high > low && cumulative + excess[order[high]] < tail) cumulative += excess[order[high--]];
    const double bin_width = std::abs((*result.frequencies)[1] - (*result.frequencies)[0]);
    const double occupied = (*result.frequencies)[order[high]] - (*result.frequencies)[order[low]] + bin_width;
    const double noise_power = floor * static_cast<double>(high - low + 1U);
    snr = {10.0 * std::log10(std::max(signal_power / std::max(noise_power, 1e-20), 1e-12)),
           3.0, "dB", Validity::unreliable, "Welch excess power over median PSD floor"};
    bandwidth = {occupied, 2.0 * bin_width, result.frequency_unit, Validity::valid,
                 "99% occupied excess power interval"};
}

}  // namespace

AnalysisResult analyze_window(
    const std::span<const std::complex<float>> input, const AnalysisConfig& config) {
    if (config.schema_version != 1U) throw std::invalid_argument("unsupported analysis schema version");
    if (input.size() < 256U) {
        AnalysisResult short_result;
        short_result.processed_samples = input.size();
        short_result.status = InferenceStatus::unknown;
        short_result.diagnostics.push_back("INSUFFICIENT_SAMPLES");
        return short_result;
    }
    for (const auto value : input) if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) throw std::invalid_argument("analysis input contains NaN or infinity");
    double power = 0.0; for (const auto value : input) power += std::norm(value); power /= input.size();
    if (!(power > 1e-16)) {
        AnalysisResult silent_result;
        silent_result.processed_samples = input.size();
        silent_result.status = InferenceStatus::unknown;
        silent_result.diagnostics.push_back("NO_MEASURABLE_POWER");
        return silent_result;
    }

    AnalysisResult result; result.processed_samples = input.size();
    spectral_estimates(input, config, result.snr, result.occupied_bandwidth);
    const auto samples = normalize(input);
    if (!config.complex_input) {
        result.carrier_offset = {std::nullopt, std::nullopt, config.sample_rate_hz ? "Hz" : "cycles/sample",
            Validity::unavailable, "carrier offset is not identifiable from an unselected real passband"};
        result.rate_candidates = estimate_rates(samples, config, 0.0);
        if (!result.rate_candidates.empty() && result.rate_candidates.front().score >= 0.025) {
            const auto best_rate = result.rate_candidates.front();
            const double scale = config.sample_rate_hz.value_or(1.0);
            result.symbol_rate = {best_rate.symbol_rate * scale, 0.025 * best_rate.symbol_rate * scale,
                config.sample_rate_hz ? "Hz" : "symbols/sample", Validity::unreliable,
                "real-signal transition periodicity; modulation inference unavailable"};
        }
        result.status = InferenceStatus::unknown;
        result.diagnostics.push_back("COMPLEX_IQ_REQUIRED_FOR_MODULATION_INFERENCE");
        return result;
    }
    const std::array<double, 3> cfos{power_cfo(samples, 2), power_cfo(samples, 4), power_cfo(samples, 8)};
    const double cfo = median({cfos[0], cfos[1], cfos[2]});
    const double scale = config.sample_rate_hz.value_or(1.0);
    result.carrier_offset = {cfo * scale, std::abs(cfos[2] - cfos[1]) * scale,
        config.sample_rate_hz ? "Hz" : "cycles/sample", Validity::unreliable,
        "median of second-, fourth-, and eighth-power carrier candidates"};
    result.rate_candidates = estimate_rates(samples, config, cfo);
    if (result.rate_candidates.empty() || result.rate_candidates.front().score < 0.025) {
        result.symbol_rate = {std::nullopt, std::nullopt,
            config.sample_rate_hz ? "Hz" : "symbols/sample", Validity::unavailable,
            "no stable transition-period candidate"};
        result.status = InferenceStatus::unknown;
        result.diagnostics.push_back("SYMBOL_RATE_UNRESOLVED");
        return result;
    }
    const auto best_rate = result.rate_candidates.front();
    result.symbol_rate = {best_rate.symbol_rate * scale, 0.025 * best_rate.symbol_rate * scale,
        config.sample_rate_hz ? "Hz" : "symbols/sample",
        best_rate.score >= 0.06 ? Validity::valid : Validity::unreliable,
        "ranked transition-energy periodicity"};
    result.modulation_candidates = classify(samples, best_rate.samples_per_symbol, cfo, best_rate.score);
    const double cp_score = delayed_correlation_peak(samples);
    double crest_factor = 0.0;
    for (const auto value : samples) crest_factor = std::max(crest_factor, std::abs(value));
    const double bandwidth_scale = config.sample_rate_hz.value_or(1.0);
    const double normalized_bandwidth = result.occupied_bandwidth.value.value_or(0.0) / bandwidth_scale;
    const bool multicarrier_like = cp_score > 0.12 && normalized_bandwidth > 0.20 && crest_factor > 3.0;
    if (multicarrier_like) {
        result.modulation_candidates.push_back({
            "OFDM/multicarrier", "UNSUPPORTED", clamp01(cp_score / 0.22),
            InferenceStatus::unsupported, 0.0, 0.0,
            {"wide occupied bandwidth", "delayed-correlation peak consistent with cyclic prefix"},
            {"outside configured single-carrier modulation set"}});
    }
    std::sort(result.modulation_candidates.begin(), result.modulation_candidates.end(),
        [](const auto& left, const auto& right) { return left.score > right.score; });

    if (!result.modulation_candidates.empty() && config.temporal_windows > 1U &&
        samples.size() / config.temporal_windows >= 256U) {
        const auto global_family = result.modulation_candidates.front().family;
        std::size_t matches = 0U;
        for (std::size_t window = 0; window < config.temporal_windows; ++window) {
            const auto begin = window * samples.size() / config.temporal_windows;
            const auto end = (window + 1U) * samples.size() / config.temporal_windows;
            std::vector<std::complex<double>> portion(samples.begin() + static_cast<std::ptrdiff_t>(begin),
                                                       samples.begin() + static_cast<std::ptrdiff_t>(end));
            auto local = classify(portion, best_rate.samples_per_symbol, cfo, best_rate.score);
            if (!local.empty() && local.front().family == global_family) ++matches;
        }
        result.temporal_consistency = static_cast<double>(matches) /
                                      static_cast<double>(config.temporal_windows);
    } else {
        result.temporal_consistency = 1.0;
    }
    for (auto& candidate : result.modulation_candidates) {
        candidate.temporal_consistency = result.temporal_consistency;
        candidate.score *= 0.8 + 0.2 * result.temporal_consistency;
    }
    std::sort(result.modulation_candidates.begin(), result.modulation_candidates.end(),
        [](const auto& left, const auto& right) { return left.score > right.score; });
    if (result.modulation_candidates.size() > config.maximum_candidates) result.modulation_candidates.resize(config.maximum_candidates);
    const double top = result.modulation_candidates.empty() ? 0.0 : result.modulation_candidates[0].score;
    const double second = result.modulation_candidates.size() < 2U ? 0.0 : result.modulation_candidates[1].score;
    double phase_increment_correlation = 0.0;
    {
        std::vector<double> frequency;
        for (std::size_t index = 1; index < samples.size(); ++index) frequency.push_back(
            std::arg(samples[index] * std::conj(samples[index - 1])) / (2.0 * pi));
        const double center = std::accumulate(frequency.begin(), frequency.end(), 0.0) / frequency.size();
        double covariance = 0.0, variance = 0.0;
        for (std::size_t index = 1; index < frequency.size(); ++index) {
            covariance += (frequency[index] - center) * (frequency[index - 1] - center);
            variance += std::pow(frequency[index] - center, 2.0);
        }
        if (variance > 0.0) phase_increment_correlation = covariance / variance;
    }
    const bool noise_like = result.snr.value.value_or(0.0) < 0.0 && phase_increment_correlation < 0.10;
    const bool coherent_low_order = !result.modulation_candidates.empty() &&
        result.modulation_candidates.front().family == "PSK" && top >= 0.40;
    if (multicarrier_like && !result.modulation_candidates.empty() &&
        result.modulation_candidates.front().family == "UNSUPPORTED") result.status = InferenceStatus::unsupported;
    else if ((noise_like && !coherent_low_order) || top < config.unknown_threshold) result.status = InferenceStatus::unknown;
    else if (top - second < config.ambiguity_margin) result.status = InferenceStatus::ambiguous;
    else result.status = InferenceStatus::candidate;
    for (auto& candidate : result.modulation_candidates) candidate.status = result.status;
    return result;
}

}  // namespace sih::estimation
