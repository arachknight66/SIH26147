#include "sih/dsp/preprocessing.hpp"
#include "sih/dsp/spectral.hpp"
#include "sih/io/recording_source.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

void check(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_near(const double actual, const double expected, const double tolerance, const char* message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        const auto raw_path = std::filesystem::temp_directory_path() / "sih_phase2_u8.iq";
        {
            const std::vector<std::uint8_t> bytes{128, 255, 0, 128};
            std::ofstream stream(raw_path, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size()));
        }
        const sih::io::RawIQSource raw(
            raw_path, sih::io::ScalarFormat::u8, sih::io::IQOrder::iq,
            sih::io::ByteOrder::little);
        check(!raw.info().sample_rate_hz.has_value(), "raw source fabricated a sample rate");
        const auto raw_chunk = raw.read_chunk(0, 16);
        check(raw_chunk.samples->size() == 2U, "raw chunk length is incorrect");
        check_near((*raw_chunk.samples)[0].real(), 0.0, 1.0e-7, "u8 centering is incorrect");
        check_near((*raw_chunk.samples)[0].imag(), 127.0 / 128.0, 1.0e-7, "u8 scaling is incorrect");
        check_near((*raw_chunk.samples)[1].real(), -1.0, 1.0e-7, "u8 negative endpoint is incorrect");
        std::filesystem::remove(raw_path);

        std::vector<std::complex<float>> samples(4096);
        for (std::size_t index = 0; index < samples.size(); ++index) {
            const auto phase = 2.0 * std::numbers::pi * 0.125 * static_cast<double>(index);
            samples[index] = std::polar(0.7F, static_cast<float>(phase)) +
                             std::complex<float>{0.15F, -0.1F};
        }
        sih::dsp::PreprocessingConfig preprocessing;
        preprocessing.remove_dc = true;
        preprocessing.enable_agc = true;
        preprocessing.mix_frequency_cycles_per_sample = 0.125;
        preprocessing.fir_taps = {0.25F, 0.5F, 0.25F};
        preprocessing.resample_up = 2;
        preprocessing.resample_down = 3;

        sih::dsp::PreprocessorSession whole_session(preprocessing);
        auto whole = whole_session.process(samples).samples;
        const auto whole_tail = whole_session.finish().samples;
        whole->insert(whole->end(), whole_tail->begin(), whole_tail->end());

        sih::dsp::PreprocessorSession chunked_session(preprocessing);
        std::vector<std::complex<float>> chunked;
        const std::vector<std::size_t> partitions{1, 7, 113, 509, 1024, 2442};
        std::size_t offset = 0;
        for (const auto length : partitions) {
            const auto part = chunked_session.process(
                std::span<const std::complex<float>>(samples).subspan(offset, length));
            chunked.insert(chunked.end(), part.samples->begin(), part.samples->end());
            offset += length;
        }
        const auto chunked_tail = chunked_session.finish().samples;
        chunked.insert(chunked.end(), chunked_tail->begin(), chunked_tail->end());
        check(offset == samples.size(), "preprocessing test partitions are invalid");
        check(chunked.size() == whole->size(), "chunked resampler changed output length");
        for (std::size_t index = 0; index < chunked.size(); ++index) {
            check_near(std::abs(chunked[index] - (*whole)[index]), 0.0, 2.0e-6,
                       "chunked preprocessing differs from whole input");
        }

        std::vector<std::complex<float>> tone(4096);
        for (std::size_t index = 0; index < tone.size(); ++index) {
            tone[index] = std::polar(
                1.0F, static_cast<float>(2.0 * std::numbers::pi * 0.125 * index));
        }
        sih::dsp::SpectralConfig spectral_config;
        spectral_config.fft_size = 256;
        spectral_config.hop_size = 128;
        spectral_config.complex_input = true;
        spectral_config.max_stft_frames = 4;
        sih::dsp::SpectralAnalyzer analyzer(spectral_config);
        analyzer.update(std::span<const std::complex<float>>(tone).first(777));
        analyzer.update(std::span<const std::complex<float>>(tone).subspan(777));
        analyzer.finish();
        const auto spectrum = analyzer.result();
        const auto peak = static_cast<std::size_t>(std::distance(
            spectrum.psd->begin(), std::max_element(spectrum.psd->begin(), spectrum.psd->end())));
        check_near((*spectrum.frequencies)[peak], 0.125, 1.0 / 256.0,
                   "normalized frequency axis or FFT peak is incorrect");
        const auto integrated = std::accumulate(spectrum.psd->begin(), spectrum.psd->end(), 0.0) / 256.0;
        check_near(integrated, 1.0, 0.02, "Welch PSD does not preserve tone power");
        check(spectrum.frequency_unit == "cycles/sample", "missing rate did not use normalized units");
        check(spectrum.stft_frames == 4U && spectrum.dropped_stft_frames > 0U,
              "STFT retention budget is not enforced");
        check(spectrum.processed_samples == tone.size(), "spectral coverage is incorrect");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
