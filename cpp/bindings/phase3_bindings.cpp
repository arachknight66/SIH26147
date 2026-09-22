#include "phase3_bindings.hpp"
#include "sih/estimation/analysis.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <span>
#include <stdexcept>

namespace py = pybind11;

namespace {
std::span<const std::complex<float>> complex_span(const py::array& input) {
    if (!input.dtype().is(py::dtype::of<std::complex<float>>())) throw py::type_error("sample input must have dtype complex64; implicit conversion is disabled");
    if (input.ndim() != 1) throw std::invalid_argument("sample input must be one-dimensional");
    if ((input.flags() & py::array::c_style) == 0) throw std::invalid_argument("sample input must be C-contiguous; implicit copies are disabled");
    return {static_cast<const std::complex<float>*>(input.data()), static_cast<std::size_t>(input.size())};
}
}

void bind_phase3(py::module_& module) {
    using namespace sih::estimation;
    py::enum_<Validity>(module, "EstimateValidity").value("VALID", Validity::valid).value("UNRELIABLE", Validity::unreliable).value("UNAVAILABLE", Validity::unavailable);
    py::enum_<InferenceStatus>(module, "InferenceStatus").value("CANDIDATE", InferenceStatus::candidate).value("AMBIGUOUS", InferenceStatus::ambiguous).value("UNKNOWN", InferenceStatus::unknown).value("UNSUPPORTED", InferenceStatus::unsupported);
    py::class_<Estimate>(module, "ParameterEstimate").def_readonly("value", &Estimate::value).def_readonly("uncertainty", &Estimate::uncertainty).def_readonly("unit", &Estimate::unit).def_readonly("validity", &Estimate::validity).def_readonly("evidence", &Estimate::evidence);
    py::class_<RateCandidate>(module, "RateCandidate").def_readonly("symbol_rate", &RateCandidate::symbol_rate).def_readonly("samples_per_symbol", &RateCandidate::samples_per_symbol).def_readonly("score", &RateCandidate::score);
    py::class_<ModulationCandidate>(module, "NativeModulationCandidate").def_readonly("label", &ModulationCandidate::label).def_readonly("family", &ModulationCandidate::family).def_readonly("score", &ModulationCandidate::score).def_readonly("status", &ModulationCandidate::status).def_readonly("constellation_error", &ModulationCandidate::constellation_error).def_readonly("temporal_consistency", &ModulationCandidate::temporal_consistency).def_readonly("evidence", &ModulationCandidate::evidence).def_readonly("contradictions", &ModulationCandidate::contradictions);
    py::class_<AnalysisConfig>(module, "AnalysisConfig").def(py::init<>()).def_readwrite("schema_version", &AnalysisConfig::schema_version).def_readwrite("sample_rate_hz", &AnalysisConfig::sample_rate_hz).def_readwrite("complex_input", &AnalysisConfig::complex_input).def_readwrite("minimum_samples_per_symbol", &AnalysisConfig::minimum_samples_per_symbol).def_readwrite("maximum_samples_per_symbol", &AnalysisConfig::maximum_samples_per_symbol).def_readwrite("maximum_candidates", &AnalysisConfig::maximum_candidates).def_readwrite("temporal_windows", &AnalysisConfig::temporal_windows).def_readwrite("unknown_threshold", &AnalysisConfig::unknown_threshold).def_readwrite("ambiguity_margin", &AnalysisConfig::ambiguity_margin);
    py::class_<AnalysisResult>(module, "WindowAnalysisResult").def_readonly("snr", &AnalysisResult::snr).def_readonly("occupied_bandwidth", &AnalysisResult::occupied_bandwidth).def_readonly("carrier_offset", &AnalysisResult::carrier_offset).def_readonly("symbol_rate", &AnalysisResult::symbol_rate).def_readonly("rate_candidates", &AnalysisResult::rate_candidates).def_readonly("modulation_candidates", &AnalysisResult::modulation_candidates).def_readonly("status", &AnalysisResult::status).def_readonly("temporal_consistency", &AnalysisResult::temporal_consistency).def_readonly("processed_samples", &AnalysisResult::processed_samples).def_readonly("diagnostics", &AnalysisResult::diagnostics);
    module.def("analyze_window", [](const py::array& input, const AnalysisConfig& config) { const auto samples = complex_span(input); py::gil_scoped_release release; return analyze_window(samples, config); }, py::arg("samples"), py::arg("config") = AnalysisConfig{});
}
