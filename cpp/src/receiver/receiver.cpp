#include "sih/receiver/receiver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sih::receiver {
namespace {

constexpr double pi = 3.14159265358979323846;

struct Constellation {
    std::vector<std::complex<double>> points;
    std::vector<std::vector<std::uint8_t>> bits;
    std::size_t bits_per_symbol{0};
    unsigned rotational_order{1};
};

unsigned gray(const unsigned value) { return value ^ (value >> 1U); }

std::vector<std::uint8_t> integer_bits(const unsigned value, const unsigned width) {
    std::vector<std::uint8_t> result(width);
    for (unsigned bit = 0; bit < width; ++bit) {
        result[bit] = static_cast<std::uint8_t>((value >> (width - bit - 1U)) & 1U);
    }
    return result;
}

Constellation make_constellation(const std::string& modulation) {
    if (modulation == "BPSK") {
        return {{{-1.0, 0.0}, {1.0, 0.0}}, {{0}, {1}}, 1U, 2U};
    }
    if (modulation == "QPSK" || modulation == "OQPSK") {
        const double scale = 1.0 / std::sqrt(2.0);
        return {{{scale, scale}, {-scale, scale}, {-scale, -scale}, {scale, -scale}},
                {{1, 1}, {0, 1}, {0, 0}, {1, 0}}, 2U, 4U};
    }
    if (modulation == "8PSK") {
        Constellation result; result.bits_per_symbol = 3U; result.rotational_order = 8U;
        for (unsigned index = 0; index < 8U; ++index) {
            result.points.push_back(std::polar(1.0, 2.0 * pi * index / 8.0));
            result.bits.push_back(integer_bits(gray(index), 3U));
        }
        return result;
    }
    unsigned side = 0U;
    if (modulation == "16-QAM") side = 4U;
    if (modulation == "64-QAM") side = 8U;
    if (modulation == "256-QAM") side = 16U;
    if (!side) return {};
    const unsigned axis_bits = static_cast<unsigned>(std::log2(side));
    Constellation result; result.bits_per_symbol = 2U * axis_bits; result.rotational_order = 4U;
    double power = 0.0;
    for (unsigned q_row = 0; q_row < side; ++q_row) {
        for (unsigned i_column = 0; i_column < side; ++i_column) {
            const double i_value = 2.0 * i_column - side + 1.0;
            const double q_value = side - 1.0 - 2.0 * q_row;
            result.points.emplace_back(i_value, q_value);
            power += i_value * i_value + q_value * q_value;
            const auto q_bits = integer_bits(gray(q_row), axis_bits);
            const auto i_bits = integer_bits(gray(i_column), axis_bits);
            std::vector<std::uint8_t> label;
            label.reserve(2U * axis_bits);
            for (unsigned bit = 0; bit < axis_bits; ++bit) {
                label.push_back(q_bits[bit]);
                label.push_back(i_bits[bit]);
            }
            result.bits.push_back(std::move(label));
        }
    }
    const double rms = std::sqrt(power / result.points.size());
    for (auto& point : result.points) point /= rms;
    return result;
}

double refined_power_cfo(const std::vector<std::complex<double>>& samples, const unsigned order) {
    std::complex<double> adjacent{0.0, 0.0};
    for (std::size_t index = 1; index < samples.size(); ++index) {
        adjacent += std::pow(samples[index], order) * std::conj(std::pow(samples[index - 1U], order));
    }
    const double coarse = std::arg(adjacent) / (2.0 * pi * order);
    double estimate = coarse;
    for (const std::size_t lag : {16U, 64U, 256U}) {
        if (samples.size() <= lag * 4U) continue;
        std::complex<double> delayed{0.0, 0.0};
        for (std::size_t index = 0; index + lag < samples.size(); ++index) {
            delayed += std::pow(samples[index + lag], order) * std::conj(std::pow(samples[index], order));
        }
        const double wrapped = std::arg(delayed) / (2.0 * pi * order * lag);
        const double spacing = 1.0 / (static_cast<double>(order) * lag);
        // Unwrap progressively.  A long-lag observation is precise but has a
        // narrow ambiguity interval, so the preceding shorter lag is its branch
        // reference.  Unwrapping every lag against the noisy adjacent estimate
        // can move a long-lag result by one whole branch.
        estimate = wrapped + std::round((estimate - wrapped) / spacing) * spacing;
    }
    return estimate;
}

double carrier_line_cfo(
    const std::vector<std::complex<double>>& samples, const unsigned order,
    const double initial) {
    std::vector<std::complex<double>> nonlinear(samples.size());
    for (std::size_t index = 0; index < samples.size(); ++index) nonlinear[index] = std::pow(samples[index], order);
    auto search = [&](const double center, const double half_width, const unsigned steps) {
        double best_frequency = center;
        double best_power = -1.0;
        for (unsigned step = 0; step <= steps; ++step) {
            const double frequency = center - half_width + 2.0 * half_width * step / steps;
            const auto rotation = std::polar(1.0, -2.0 * pi * order * frequency);
            std::complex<double> oscillator{1.0, 0.0};
            std::complex<double> total{0.0, 0.0};
            for (const auto value : nonlinear) { total += value * oscillator; oscillator *= rotation; }
            const double power = std::norm(total);
            if (power > best_power) { best_power = power; best_frequency = frequency; }
        }
        return best_frequency;
    };
    // The nonlinear spectrum contains data-dependent sidelobes.  Searching the
    // complete ambiguity interval can therefore select a stronger sidelobe than
    // the carrier line on short, pulse-shaped captures.  The delayed-product
    // estimate supplies the ambiguity branch; refine only inside its capture
    // neighbourhood and retain the other branches as explicit ambiguities.
    const double half_width = std::min(0.005, 0.45 / static_cast<double>(order));
    const double coarse = search(initial, half_width, 320U);
    return search(coarse, half_width / 160.0, 80U);
}

std::vector<double> rrc_taps(const double sps, const double alpha) {
    const auto half = static_cast<int>(std::ceil(6.0 * sps));
    std::vector<double> taps;
    taps.reserve(2U * half + 1U);
    for (int index = -half; index <= half; ++index) {
        const double t = static_cast<double>(index) / sps;
        double value = 0.0;
        if (std::abs(t) < 1e-12) {
            value = 1.0 - alpha + 4.0 * alpha / pi;
        } else if (std::abs(std::abs(4.0 * alpha * t) - 1.0) < 1e-9) {
            value = alpha / std::sqrt(2.0) * ((1.0 + 2.0 / pi) * std::sin(pi / (4.0 * alpha)) +
                    (1.0 - 2.0 / pi) * std::cos(pi / (4.0 * alpha)));
        } else {
            value = (std::sin(pi * t * (1.0 - alpha)) + 4.0 * alpha * t *
                    std::cos(pi * t * (1.0 + alpha))) /
                    (pi * t * (1.0 - std::pow(4.0 * alpha * t, 2.0)));
        }
        taps.push_back(value);
    }
    const double energy = std::inner_product(taps.begin(), taps.end(), taps.begin(), 0.0);
    for (auto& tap : taps) tap /= std::sqrt(energy);
    return taps;
}

std::vector<std::complex<double>> filter(
    const std::vector<std::complex<double>>& samples, const std::vector<double>& taps) {
    std::vector<std::complex<double>> result(samples.size(), {0.0, 0.0});
    const auto half = taps.size() / 2U;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        for (std::size_t tap = 0; tap < taps.size(); ++tap) {
            const auto source = static_cast<std::ptrdiff_t>(index) + static_cast<std::ptrdiff_t>(tap) -
                                static_cast<std::ptrdiff_t>(half);
            if (source >= 0 && source < static_cast<std::ptrdiff_t>(samples.size())) {
                result[index] += samples[static_cast<std::size_t>(source)] * taps[tap];
            }
        }
    }
    return result;
}

std::complex<double> interpolate(const std::vector<std::complex<double>>& samples, const double position) {
    if (position <= 0.0) return samples.front();
    const auto lower = static_cast<std::size_t>(position);
    if (lower + 1U >= samples.size()) return samples.back();
    const double fraction = position - static_cast<double>(lower);
    return samples[lower] * (1.0 - fraction) + samples[lower + 1U] * fraction;
}

struct Fit {
    double error{std::numeric_limits<double>::infinity()};
    double timing{0.0};
    double phase{0.0};
    double scale{1.0};
    std::vector<std::complex<double>> symbols;
    std::vector<double> offsets;
};

Fit fit_signal(
    const std::vector<std::complex<double>>& samples, const Constellation& constellation,
    const ReceiverConfig& config, const double cfo) {
    Fit best;
    const double timing_step = 0.125;
    const double phase_span = 2.0 * pi / constellation.rotational_order;
    const unsigned phase_steps = config.phase_reference_radians ? 1U : 48U;
    for (double timing = 0.0; timing < config.samples_per_symbol; timing += timing_step) {
        std::vector<std::complex<double>> symbols;
        std::vector<double> offsets;
        for (double position = timing; position < samples.size(); position += config.samples_per_symbol) {
            symbols.push_back(interpolate(samples, position) * std::polar(1.0, -2.0 * pi * cfo * position));
            offsets.push_back(position);
        }
        if (symbols.size() < 32U) continue;
        double power = 0.0; for (const auto value : symbols) power += std::norm(value);
        const double scale = std::sqrt(power / symbols.size());
        for (unsigned phase_step = 0; phase_step < phase_steps; ++phase_step) {
            const double phase = config.phase_reference_radians.value_or(phase_span * phase_step / phase_steps);
            const auto rotation = std::polar(1.0, -phase);
            double error = 0.0;
            const std::size_t stride = std::max<std::size_t>(1U, symbols.size() / 2048U);
            std::size_t used = 0U;
            for (std::size_t index = config.acquisition_symbols; index < symbols.size(); index += stride) {
                const auto value = symbols[index] * rotation / scale;
                double nearest = std::numeric_limits<double>::infinity();
                for (const auto point : constellation.points) nearest = std::min(nearest, std::norm(value - point));
                error += nearest; ++used;
            }
            error /= std::max<std::size_t>(used, 1U);
            if (error < best.error) best = {error, timing, phase, scale, symbols, offsets};
        }
    }
    return best;
}

ReceiverResult demodulate_linear(
    const std::vector<std::complex<double>>& raw, const ReceiverConfig& config,
    const Constellation& constellation) {
    ReceiverResult result;
    result.symbols = std::make_shared<std::vector<std::complex<float>>>();
    result.hard_bits = std::make_shared<std::vector<std::uint8_t>>();
    result.soft_llrs = std::make_shared<std::vector<float>>();
    result.sample_offsets = std::make_shared<std::vector<double>>();
    result.modulation = config.modulation;
    result.bits_per_symbol = constellation.bits_per_symbol;
    const double coarse_cfo = config.coarse_cfo_cycles_per_sample.value_or(
        refined_power_cfo(raw, constellation.rotational_order));
    const double cfo = config.carrier_reference_cycles_per_sample.value_or(
        carrier_line_cfo(raw, constellation.rotational_order, coarse_cfo));
    auto samples = raw;
    auto fit = fit_signal(samples, constellation, config, cfo);
    if (config.pulse_shape == "rrc" || config.pulse_shape == "auto") {
        auto filtered = filter(raw, rrc_taps(config.samples_per_symbol, config.rrc_rolloff));
        auto filtered_fit = fit_signal(filtered, constellation, config, cfo);
        if (config.pulse_shape == "rrc" || filtered_fit.error < fit.error) {
            samples = std::move(filtered);
            fit = std::move(filtered_fit);
        }
    }
    if (!std::isfinite(fit.error) || fit.symbols.size() <= config.acquisition_symbols) {
        result.acquisition_status = AcquisitionStatus::insufficient_data;
        result.diagnostics.push_back({core::Severity::warning, "RECEIVER_INSUFFICIENT_DATA",
            "Not enough symbols remained after acquisition.", ""});
        return result;
    }
    result.carrier_offset = cfo * config.sample_rate_hz.value_or(1.0);
    result.carrier_offset_unit = config.sample_rate_hz ? "Hz" : "cycles/sample";
    result.timing_offset_samples = fit.timing;
    result.timing_error = fit.error;
    result.carrier_error = fit.error;
    result.evm_percent = 100.0 * std::sqrt(fit.error);
    result.noise_variance = config.noise_variance.value_or(std::max(fit.error, 1e-8));
    const auto rotation = std::polar(1.0, -fit.phase);
    for (std::size_t symbol_index = config.acquisition_symbols; symbol_index < fit.symbols.size(); ++symbol_index) {
        const auto value = fit.symbols[symbol_index] * rotation / fit.scale;
        std::size_t nearest_index = 0U;
        double nearest = std::numeric_limits<double>::infinity();
        for (std::size_t point = 0; point < constellation.points.size(); ++point) {
            const double distance = std::norm(value - constellation.points[point]);
            if (distance < nearest) { nearest = distance; nearest_index = point; }
        }
        result.symbols->emplace_back(value);
        result.sample_offsets->push_back(fit.offsets[symbol_index]);
        result.hard_bits->insert(result.hard_bits->end(), constellation.bits[nearest_index].begin(),
                                 constellation.bits[nearest_index].end());
        for (std::size_t bit = 0; bit < constellation.bits_per_symbol; ++bit) {
            double distance_zero = std::numeric_limits<double>::infinity();
            double distance_one = std::numeric_limits<double>::infinity();
            for (std::size_t point = 0; point < constellation.points.size(); ++point) {
                auto& target = constellation.bits[point][bit] ? distance_one : distance_zero;
                target = std::min(target, std::norm(value - constellation.points[point]));
            }
            result.soft_llrs->push_back(static_cast<float>((distance_zero - distance_one) / result.noise_variance));
        }
    }
    const double lock_limit = constellation.points.size() >= 64U ? 0.16 : 0.12;
    result.acquisition_status = fit.error < lock_limit ? AcquisitionStatus::locked : AcquisitionStatus::unlocked;
    result.mapping_status = config.phase_reference_radians ? core::EvidenceStatus::verified : core::EvidenceStatus::unverified;
    if (!config.phase_reference_radians) {
        for (unsigned rotation_index = 0; rotation_index < constellation.rotational_order; ++rotation_index) {
            result.unresolved_phase_rotations.push_back(2.0 * pi * rotation_index / constellation.rotational_order);
        }
        result.diagnostics.push_back({core::Severity::info, "PHASE_AMBIGUITY_UNRESOLVED",
            "Carrier lock does not determine the absolute symbol mapping.",
            std::to_string(constellation.rotational_order) + " rotational mappings remain."});
    }
    if (!config.carrier_reference_cycles_per_sample) {
        const double spacing = 1.0 / static_cast<double>(constellation.rotational_order);
        for (int alias = -4; alias <= 4; ++alias) {
            const double candidate = cfo + alias * spacing;
            if (candidate >= -0.5 && candidate < 0.5) result.unresolved_carrier_offsets.push_back(
                candidate * config.sample_rate_hz.value_or(1.0));
        }
        if (result.unresolved_carrier_offsets.size() > 1U) result.diagnostics.push_back({
            core::Severity::info, "CARRIER_ALIAS_UNRESOLVED",
            "Nonlinear carrier acquisition leaves frequency aliases without external band evidence.",
            std::to_string(result.unresolved_carrier_offsets.size()) + " aliases remain."});
    }
    return result;
}

ReceiverResult unsupported_result(const ReceiverConfig& config) {
    ReceiverResult result;
    result.symbols = std::make_shared<std::vector<std::complex<float>>>();
    result.hard_bits = std::make_shared<std::vector<std::uint8_t>>();
    result.soft_llrs = std::make_shared<std::vector<float>>();
    result.sample_offsets = std::make_shared<std::vector<double>>();
    result.modulation = config.modulation;
    result.acquisition_status = AcquisitionStatus::unsupported;
    result.diagnostics.push_back({core::Severity::warning, "RECEIVER_PROFILE_UNSUPPORTED",
        "The requested receiver profile is not implemented in the native core.", config.modulation});
    return result;
}

ReceiverResult demodulate_fsk(const std::vector<std::complex<double>>& samples, const ReceiverConfig& config) {
    unsigned states = 0U;
    if (config.modulation == "2-FSK") states = 2U;
    if (config.modulation == "4-FSK") states = 4U;
    if (config.modulation == "8-FSK") states = 8U;
    if (config.modulation == "MSK") states = 2U;
    if (!states) return unsupported_result(config);
    ReceiverResult result;
    result.symbols = std::make_shared<std::vector<std::complex<float>>>();
    result.hard_bits = std::make_shared<std::vector<std::uint8_t>>();
    result.soft_llrs = std::make_shared<std::vector<float>>();
    result.sample_offsets = std::make_shared<std::vector<double>>();
    result.modulation = config.modulation;
    result.bits_per_symbol = static_cast<std::size_t>(std::log2(states));
    std::vector<double> frequency(samples.size() - 1U);
    for (std::size_t index = 1; index < samples.size(); ++index) frequency[index - 1U] =
        std::arg(samples[index] * std::conj(samples[index - 1U])) / (2.0 * pi);
    auto sorted = frequency; std::sort(sorted.begin(), sorted.end());
    std::vector<double> centers(states);
    for (unsigned state = 0; state < states; ++state) centers[state] = sorted[(2U * state + 1U) * sorted.size() / (2U * states)];
    for (unsigned iteration = 0; iteration < 30U; ++iteration) {
        std::vector<double> sums(states, 0.0); std::vector<std::size_t> counts(states, 0U);
        for (const auto value : frequency) {
            unsigned best = 0U;
            for (unsigned state = 1; state < states; ++state) if (std::abs(value - centers[state]) < std::abs(value - centers[best])) best = state;
            sums[best] += value; ++counts[best];
        }
        for (unsigned state = 0; state < states; ++state) if (counts[state]) centers[state] = sums[state] / counts[state];
    }
    std::sort(centers.begin(), centers.end());
    double best_error = std::numeric_limits<double>::infinity(); double best_timing = 0.0;
    std::vector<double> best_metrics;
    for (double timing = 0.0; timing < config.samples_per_symbol; timing += 0.125) {
        std::vector<double> metrics;
        for (double position = timing; position + config.samples_per_symbol <= frequency.size(); position += config.samples_per_symbol) {
            const auto begin = static_cast<std::size_t>(position);
            const auto end = std::min(frequency.size(), static_cast<std::size_t>(position + config.samples_per_symbol));
            metrics.push_back(std::accumulate(frequency.begin() + begin, frequency.begin() + end, 0.0) /
                              static_cast<double>(end - begin));
        }
        double error = 0.0;
        for (const auto metric : metrics) {
            double nearest = std::numeric_limits<double>::infinity();
            for (const auto center : centers) nearest = std::min(nearest, std::pow(metric - center, 2.0));
            error += nearest;
        }
        error /= std::max<std::size_t>(metrics.size(), 1U);
        if (error < best_error) { best_error = error; best_timing = timing; best_metrics = std::move(metrics); }
    }
    const double noise = config.noise_variance.value_or(std::max(best_error, 1e-8));
    for (std::size_t index = config.acquisition_symbols; index < best_metrics.size(); ++index) {
        const double metric = best_metrics[index];
        unsigned nearest = 0U;
        for (unsigned state = 1U; state < states; ++state) if (std::abs(metric - centers[state]) < std::abs(metric - centers[nearest])) nearest = state;
        const auto bits = integer_bits(gray(nearest), static_cast<unsigned>(result.bits_per_symbol));
        result.hard_bits->insert(result.hard_bits->end(), bits.begin(), bits.end());
        result.symbols->emplace_back(metric, 0.0);
        result.sample_offsets->push_back(best_timing + index * config.samples_per_symbol);
        for (std::size_t bit = 0; bit < result.bits_per_symbol; ++bit) {
            double d0 = std::numeric_limits<double>::infinity(), d1 = d0;
            for (unsigned state = 0; state < states; ++state) {
                auto& target = integer_bits(gray(state), static_cast<unsigned>(result.bits_per_symbol))[bit] ? d1 : d0;
                target = std::min(target, std::pow(metric - centers[state], 2.0));
            }
            result.soft_llrs->push_back(static_cast<float>((d0 - d1) / noise));
        }
    }
    result.timing_offset_samples = best_timing;
    result.timing_error = best_error;
    result.carrier_error = best_error;
    result.evm_percent = 100.0 * std::sqrt(best_error) /
        std::max(centers.back() - centers.front(), 1e-8);
    result.noise_variance = noise;
    result.carrier_offset = 0.5 * (centers.front() + centers.back()) * config.sample_rate_hz.value_or(1.0);
    result.carrier_offset_unit = config.sample_rate_hz ? "Hz" : "cycles/sample";
    result.acquisition_status = best_metrics.size() > config.acquisition_symbols + 16U &&
        result.evm_percent < 25.0 ? AcquisitionStatus::locked : AcquisitionStatus::unlocked;
    result.mapping_status = core::EvidenceStatus::unverified;
    result.diagnostics.push_back({core::Severity::info, "FSK_TONE_MAPPING_UNVERIFIED",
        "Tone order was mapped to ascending Gray-coded symbols but lacks frame-level confirmation.", ""});
    return result;
}

ReceiverResult demodulate_differential(
    const std::vector<std::complex<double>>& samples, const ReceiverConfig& config) {
    const bool binary = config.modulation == "DBPSK";
    ReceiverConfig base = config;
    base.modulation = binary ? "BPSK" : "QPSK";
    if (base.acquisition_symbols > 0U) --base.acquisition_symbols;
    auto absolute = demodulate_linear(samples, base, make_constellation(base.modulation));
    absolute.modulation = config.modulation;
    absolute.bits_per_symbol = binary ? 1U : 2U;
    auto differential_bits = std::make_shared<std::vector<std::uint8_t>>();
    auto differential_llrs = std::make_shared<std::vector<float>>();
    auto differential_symbols = std::make_shared<std::vector<std::complex<float>>>();
    auto differential_offsets = std::make_shared<std::vector<double>>();
    if (absolute.symbols->size() >= 2U) {
        for (std::size_t index = 1; index < absolute.symbols->size(); ++index) {
            auto value = static_cast<std::complex<double>>((*absolute.symbols)[index]) *
                         std::conj(static_cast<std::complex<double>>((*absolute.symbols)[index - 1U]));
            value /= std::max(std::abs(value), 1e-12);
            differential_symbols->emplace_back(value);
            differential_offsets->push_back((*absolute.sample_offsets)[index]);
            if (binary) {
                const std::uint8_t bit = value.real() < 0.0 ? 1U : 0U;
                differential_bits->push_back(bit);
                differential_llrs->push_back(static_cast<float>(-2.0 * value.real() /
                    std::max(absolute.noise_variance, 1e-8)));
            } else {
                int phase_index = static_cast<int>(std::llround(std::arg(value) / (pi / 2.0)));
                phase_index = (phase_index % 4 + 4) % 4;
                const auto bits = integer_bits(gray(static_cast<unsigned>(phase_index)), 2U);
                differential_bits->insert(differential_bits->end(), bits.begin(), bits.end());
                for (unsigned bit = 0; bit < 2U; ++bit) {
                    double d0 = std::numeric_limits<double>::infinity(), d1 = d0;
                    for (unsigned state = 0; state < 4U; ++state) {
                        const auto labels = integer_bits(gray(state), 2U);
                        auto& target = labels[bit] ? d1 : d0;
                        target = std::min(target, std::norm(value - std::polar(1.0, state * pi / 2.0)));
                    }
                    differential_llrs->push_back(static_cast<float>((d0 - d1) /
                        std::max(absolute.noise_variance, 1e-8)));
                }
            }
        }
    }
    absolute.symbols = std::move(differential_symbols);
    absolute.sample_offsets = std::move(differential_offsets);
    absolute.hard_bits = std::move(differential_bits);
    absolute.soft_llrs = std::move(differential_llrs);
    absolute.unresolved_phase_rotations.clear();
    absolute.diagnostics.push_back({core::Severity::info, "DIFFERENTIAL_PHASE_RESOLVED",
        "Differential decoding removed constant carrier-phase ambiguity.", ""});
    return absolute;
}

}  // namespace

ReceiverResult demodulate(
    const std::span<const std::complex<float>> input, const ReceiverConfig& config) {
    if (config.schema_version != 1U) throw std::invalid_argument("unsupported receiver schema version");
    if (!(config.samples_per_symbol >= 2.0) || !std::isfinite(config.samples_per_symbol)) throw std::invalid_argument("samples_per_symbol must be finite and at least two");
    std::vector<std::complex<double>> samples; samples.reserve(input.size());
    std::complex<double> mean{0.0, 0.0};
    for (const auto value : input) {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) throw std::invalid_argument("receiver input contains NaN or infinity");
        mean += static_cast<std::complex<double>>(value);
    }
    if (!input.empty()) mean /= static_cast<double>(input.size());
    double power = 0.0;
    for (const auto value : input) { const auto centered = static_cast<std::complex<double>>(value) - mean; samples.push_back(centered); power += std::norm(centered); }
    if (input.size() < 64U || power <= 1e-16) {
        auto result = unsupported_result(config);
        result.acquisition_status = AcquisitionStatus::insufficient_data;
        result.diagnostics.front().code = "RECEIVER_INSUFFICIENT_DATA";
        result.diagnostics.front().message = "At least 64 non-silent samples are required.";
        return result;
    }
    if (config.modulation == "DBPSK" || config.modulation == "DQPSK") {
        return demodulate_differential(samples, config);
    }
    const auto constellation = make_constellation(config.modulation);
    if (!constellation.points.empty()) return demodulate_linear(samples, config, constellation);
    if (config.modulation.find("FSK") != std::string::npos || config.modulation == "MSK") {
        return demodulate_fsk(samples, config);
    }
    return unsupported_result(config);
}

ReceiverSession::ReceiverSession(ReceiverConfig config) : config_(std::move(config)) {}

std::size_t ReceiverSession::process(const std::span<const std::complex<float>> samples) {
    if (flushed_) throw std::logic_error("receiver session has already been flushed");
    if (samples_.size() + samples.size() > config_.maximum_buffered_samples) throw std::length_error("receiver acquisition buffer limit exceeded");
    samples_.insert(samples_.end(), samples.begin(), samples.end());
    return samples.size();
}

ReceiverResult ReceiverSession::flush() {
    if (flushed_) throw std::logic_error("receiver session has already been flushed");
    flushed_ = true;
    return demodulate(samples_, config_);
}

void ReceiverSession::reset() { samples_.clear(); flushed_ = false; }
std::size_t ReceiverSession::buffered_samples() const noexcept { return samples_.size(); }

}  // namespace sih::receiver
