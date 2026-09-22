#include "phase2_bindings.hpp"

#include "sih/dsp/preprocessing.hpp"
#include "sih/dsp/spectral.hpp"
#include "sih/io/recording_source.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <complex>
#include <memory>
#include <span>
#include <vector>

namespace py = pybind11;

namespace {

template <typename Value>
py::array vector_view(const std::shared_ptr<std::vector<Value>>& values) {
    auto* owner = new std::shared_ptr<std::vector<Value>>(values);
    py::capsule capsule(owner, [](void* pointer) {
        delete static_cast<std::shared_ptr<std::vector<Value>>*>(pointer);
    });
    return py::array(
        py::dtype::of<Value>(),
        {static_cast<py::ssize_t>((*owner)->size())},
        {static_cast<py::ssize_t>(sizeof(Value))},
        (*owner)->data(),
        capsule);
}

py::array stft_view(const sih::dsp::SpectralResult& result) {
    auto* owner = new std::shared_ptr<std::vector<float>>(result.stft_power);
    py::capsule capsule(owner, [](void* pointer) {
        delete static_cast<std::shared_ptr<std::vector<float>>*>(pointer);
    });
    return py::array(
        py::dtype::of<float>(),
        {static_cast<py::ssize_t>(result.stft_frames),
         static_cast<py::ssize_t>(result.frequency_bins)},
        {static_cast<py::ssize_t>(result.frequency_bins * sizeof(float)),
         static_cast<py::ssize_t>(sizeof(float))},
        (*owner)->data(),
        capsule);
}

std::span<const std::complex<float>> complex_span(const py::array& input) {
    if (!input.dtype().is(py::dtype::of<std::complex<float>>())) {
        throw py::type_error("sample input must have dtype complex64; implicit conversion is disabled");
    }
    if (input.ndim() != 1) {
        throw std::invalid_argument("sample input must be a one-dimensional complex64 array");
    }
    if ((input.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument("sample input must be C-contiguous; implicit copies are disabled");
    }
    return {static_cast<const std::complex<float>*>(input.data()),
            static_cast<std::size_t>(input.size())};
}

}  // namespace

void bind_phase2(py::module_& module) {
    using namespace sih;

    py::enum_<io::ScalarFormat>(module, "ScalarFormat")
        .value("COMPLEX_F32", io::ScalarFormat::complex_f32)
        .value("F32", io::ScalarFormat::f32)
        .value("S8", io::ScalarFormat::s8)
        .value("U8", io::ScalarFormat::u8)
        .value("S16", io::ScalarFormat::s16)
        .value("S24", io::ScalarFormat::s24)
        .value("S32", io::ScalarFormat::s32);
    py::enum_<io::ByteOrder>(module, "ByteOrder")
        .value("LITTLE", io::ByteOrder::little)
        .value("BIG", io::ByteOrder::big);
    py::enum_<io::IQOrder>(module, "IQOrder")
        .value("IQ", io::IQOrder::iq)
        .value("QI", io::IQOrder::qi);
    py::enum_<io::WavChannelMode>(module, "WavChannelMode")
        .value("MONO", io::WavChannelMode::mono)
        .value("CHANNEL_0", io::WavChannelMode::channel_0)
        .value("CHANNEL_1", io::WavChannelMode::channel_1)
        .value("STEREO_IQ", io::WavChannelMode::stereo_iq);

    py::class_<io::SourceInfo>(module, "SourceInfo")
        .def_readonly("sample_count", &io::SourceInfo::sample_count)
        .def_readonly("channel_count", &io::SourceInfo::channel_count)
        .def_readonly("scalar_format", &io::SourceInfo::scalar_format)
        .def_readonly("sample_rate_hz", &io::SourceInfo::sample_rate_hz)
        .def_readonly("complex_samples", &io::SourceInfo::complex_samples)
        .def_readonly("normalized", &io::SourceInfo::normalized)
        .def_readonly("conversion", &io::SourceInfo::conversion);
    py::class_<io::SampleChunk>(module, "SampleChunk")
        .def_property_readonly("samples", [](const io::SampleChunk& value) {
            return vector_view(value.samples);
        })
        .def_readonly("start_sample", &io::SampleChunk::start_sample)
        .def_readonly("source_sample_count", &io::SampleChunk::source_sample_count);
    py::class_<io::RawIQSource>(module, "RawIQSource")
        .def(py::init<std::filesystem::path, io::ScalarFormat, io::IQOrder,
                      io::ByteOrder, std::optional<double>>(),
             py::arg("path"), py::arg("format"), py::arg("iq_order"),
             py::arg("byte_order"), py::arg("sample_rate_hz") = std::nullopt)
        .def_property_readonly("info", &io::RawIQSource::info,
                               py::return_value_policy::reference_internal)
        .def("read_chunk", [](const io::RawIQSource& source,
                              const std::uint64_t start, const std::size_t count) {
            py::gil_scoped_release release;
            return source.read_chunk(start, count);
        });
    py::class_<io::WavSource>(module, "WavSource")
        .def(py::init<std::filesystem::path, io::WavChannelMode>())
        .def_property_readonly("info", &io::WavSource::info,
                               py::return_value_policy::reference_internal)
        .def("read_chunk", [](const io::WavSource& source,
                              const std::uint64_t start, const std::size_t count) {
            py::gil_scoped_release release;
            return source.read_chunk(start, count);
        });

    py::class_<dsp::PreprocessingConfig>(module, "PreprocessingConfig")
        .def(py::init<>())
        .def_readwrite("schema_version", &dsp::PreprocessingConfig::schema_version)
        .def_readwrite("remove_dc", &dsp::PreprocessingConfig::remove_dc)
        .def_readwrite("dc_pole", &dsp::PreprocessingConfig::dc_pole)
        .def_readwrite("enable_agc", &dsp::PreprocessingConfig::enable_agc)
        .def_readwrite("agc_target_rms", &dsp::PreprocessingConfig::agc_target_rms)
        .def_readwrite("agc_smoothing", &dsp::PreprocessingConfig::agc_smoothing)
        .def_readwrite("mix_frequency_cycles_per_sample",
                       &dsp::PreprocessingConfig::mix_frequency_cycles_per_sample)
        .def_readwrite("fir_taps", &dsp::PreprocessingConfig::fir_taps)
        .def_readwrite("resample_up", &dsp::PreprocessingConfig::resample_up)
        .def_readwrite("resample_down", &dsp::PreprocessingConfig::resample_down);
    module.def("design_lowpass_fir", &dsp::design_lowpass_fir,
               py::arg("cutoff_cycles_per_sample"), py::arg("tap_count"));
    module.def("design_bandpass_fir", &dsp::design_bandpass_fir,
               py::arg("lower_cycles_per_sample"),
               py::arg("upper_cycles_per_sample"), py::arg("tap_count"));
    py::class_<dsp::PreprocessingChunk>(module, "PreprocessingChunk")
        .def_property_readonly("samples", [](const dsp::PreprocessingChunk& value) {
            return vector_view(value.samples);
        })
        .def_readonly("input_samples_consumed", &dsp::PreprocessingChunk::input_samples_consumed)
        .def_readonly("output_samples_produced", &dsp::PreprocessingChunk::output_samples_produced);
    py::class_<dsp::PreprocessorSession>(module, "PreprocessorSession")
        .def(py::init<dsp::PreprocessingConfig>())
        .def("process", [](dsp::PreprocessorSession& session,
                           const py::array& input) {
            const auto samples = complex_span(input);
            py::gil_scoped_release release;
            return session.process(samples);
        })
        .def("finish", &dsp::PreprocessorSession::finish)
        .def("reset", &dsp::PreprocessorSession::reset);
    py::class_<dsp::TimeStatistics>(module, "NativeTimeStatistics")
        .def_readonly("sample_count", &dsp::TimeStatistics::sample_count)
        .def_readonly("mean_i", &dsp::TimeStatistics::mean_i)
        .def_readonly("mean_q", &dsp::TimeStatistics::mean_q)
        .def_readonly("variance_i", &dsp::TimeStatistics::variance_i)
        .def_readonly("variance_q", &dsp::TimeStatistics::variance_q)
        .def_readonly("rms_amplitude", &dsp::TimeStatistics::rms_amplitude)
        .def_readonly("peak_amplitude", &dsp::TimeStatistics::peak_amplitude)
        .def_readonly("crest_factor", &dsp::TimeStatistics::crest_factor)
        .def_readonly("clipping_available", &dsp::TimeStatistics::clipping_available)
        .def_readonly("clipping_fraction", &dsp::TimeStatistics::clipping_fraction);
    py::class_<dsp::StatisticsAccumulator>(module, "StatisticsAccumulator")
        .def(py::init<float>(), py::arg("clipping_level") = 0.999F)
        .def("update", [](dsp::StatisticsAccumulator& accumulator,
                          const py::array& input) {
            const auto samples = complex_span(input);
            py::gil_scoped_release release;
            accumulator.update(samples);
        })
        .def("result", &dsp::StatisticsAccumulator::result)
        .def("reset", &dsp::StatisticsAccumulator::reset);

    py::class_<dsp::SpectralConfig>(module, "SpectralConfig")
        .def(py::init<>())
        .def_readwrite("schema_version", &dsp::SpectralConfig::schema_version)
        .def_readwrite("fft_size", &dsp::SpectralConfig::fft_size)
        .def_readwrite("hop_size", &dsp::SpectralConfig::hop_size)
        .def_readwrite("complex_input", &dsp::SpectralConfig::complex_input)
        .def_readwrite("sample_rate_hz", &dsp::SpectralConfig::sample_rate_hz)
        .def_readwrite("max_stft_frames", &dsp::SpectralConfig::max_stft_frames);
    py::class_<dsp::SpectralResult>(module, "NativeSpectralResult")
        .def_property_readonly("frequencies", [](const dsp::SpectralResult& value) {
            return vector_view(value.frequencies);
        })
        .def_property_readonly("psd", [](const dsp::SpectralResult& value) {
            return vector_view(value.psd);
        })
        .def_property_readonly("times", [](const dsp::SpectralResult& value) {
            return vector_view(value.times);
        })
        .def_property_readonly("stft_power", &stft_view)
        .def_readonly("frequency_bins", &dsp::SpectralResult::frequency_bins)
        .def_readonly("stft_frames", &dsp::SpectralResult::stft_frames)
        .def_readonly("processed_samples", &dsp::SpectralResult::processed_samples)
        .def_readonly("welch_segments", &dsp::SpectralResult::welch_segments)
        .def_readonly("dropped_stft_frames", &dsp::SpectralResult::dropped_stft_frames)
        .def_readonly("frequency_unit", &dsp::SpectralResult::frequency_unit)
        .def_readonly("time_unit", &dsp::SpectralResult::time_unit)
        .def_readonly("power_unit", &dsp::SpectralResult::power_unit);
    py::class_<dsp::SpectralAnalyzer>(module, "SpectralAnalyzer")
        .def(py::init<dsp::SpectralConfig>())
        .def("update", [](dsp::SpectralAnalyzer& analyzer,
                          const py::array& input) {
            const auto samples = complex_span(input);
            py::gil_scoped_release release;
            analyzer.update(samples);
        })
        .def("finish", &dsp::SpectralAnalyzer::finish, py::arg("include_partial") = true)
        .def("result", &dsp::SpectralAnalyzer::result)
        .def("reset", &dsp::SpectralAnalyzer::reset);
    py::class_<dsp::SpectralBand>(module, "SpectralBand")
        .def_readonly("lower_frequency", &dsp::SpectralBand::lower_frequency)
        .def_readonly("upper_frequency", &dsp::SpectralBand::upper_frequency)
        .def_readonly("center_frequency", &dsp::SpectralBand::center_frequency)
        .def_readonly("occupied_bandwidth", &dsp::SpectralBand::occupied_bandwidth)
        .def_readonly("integrated_power", &dsp::SpectralBand::integrated_power)
        .def_readonly("peak_to_noise_db", &dsp::SpectralBand::peak_to_noise_db);
    module.def(
        "detect_spectral_bands",
        [](const py::array& frequencies,
           const py::array& psd,
           const double threshold_above_noise_db,
           const std::size_t minimum_bins) {
            if (!frequencies.dtype().is(py::dtype::of<double>()) ||
                !psd.dtype().is(py::dtype::of<float>())) {
                throw py::type_error("frequencies must be float64 and PSD must be float32");
            }
            if (frequencies.ndim() != 1 || psd.ndim() != 1) {
                throw std::invalid_argument("frequencies and PSD must be one-dimensional arrays");
            }
            if ((frequencies.flags() & py::array::c_style) == 0 ||
                (psd.flags() & py::array::c_style) == 0) {
                throw std::invalid_argument("frequencies and PSD must be C-contiguous");
            }
            py::gil_scoped_release release;
            return dsp::detect_spectral_bands(
                {static_cast<const double*>(frequencies.data()),
                 static_cast<std::size_t>(frequencies.size())},
                {static_cast<const float*>(psd.data()),
                 static_cast<std::size_t>(psd.size())},
                threshold_above_noise_db,
                minimum_bins);
        },
        py::arg("frequencies"), py::arg("psd"),
        py::arg("threshold_above_noise_db") = 8.0,
        py::arg("minimum_bins") = 2);
}
