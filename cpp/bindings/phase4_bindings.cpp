#include "phase4_bindings.hpp"
#include "sih/receiver/receiver.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <memory>
#include <span>
#include <stdexcept>

namespace py = pybind11;
namespace {
template <typename Value>
py::array vector_view(const std::shared_ptr<std::vector<Value>>& values) {
    auto* owner = new std::shared_ptr<std::vector<Value>>(values);
    py::capsule capsule(owner, [](void* pointer) { delete static_cast<std::shared_ptr<std::vector<Value>>*>(pointer); });
    return py::array(py::dtype::of<Value>(), {static_cast<py::ssize_t>((*owner)->size())},
                     {static_cast<py::ssize_t>(sizeof(Value))}, (*owner)->data(), capsule);
}
std::span<const std::complex<float>> complex_span(const py::array& input) {
    if (!input.dtype().is(py::dtype::of<std::complex<float>>())) throw py::type_error("sample input must have dtype complex64; implicit conversion is disabled");
    if (input.ndim() != 1) throw std::invalid_argument("sample input must be one-dimensional");
    if ((input.flags() & py::array::c_style) == 0) throw std::invalid_argument("sample input must be C-contiguous; implicit copies are disabled");
    return {static_cast<const std::complex<float>*>(input.data()), static_cast<std::size_t>(input.size())};
}
}

void bind_phase4(py::module_& module) {
    using namespace sih::receiver;
    py::enum_<AcquisitionStatus>(module, "AcquisitionStatus").value("LOCKED", AcquisitionStatus::locked).value("UNLOCKED", AcquisitionStatus::unlocked).value("UNSUPPORTED", AcquisitionStatus::unsupported).value("INSUFFICIENT_DATA", AcquisitionStatus::insufficient_data);
    py::class_<ReceiverConfig>(module, "ReceiverConfig").def(py::init<>()).def_readwrite("schema_version", &ReceiverConfig::schema_version).def_readwrite("modulation", &ReceiverConfig::modulation).def_readwrite("samples_per_symbol", &ReceiverConfig::samples_per_symbol).def_readwrite("sample_rate_hz", &ReceiverConfig::sample_rate_hz).def_readwrite("coarse_cfo_cycles_per_sample", &ReceiverConfig::coarse_cfo_cycles_per_sample).def_readwrite("carrier_reference_cycles_per_sample", &ReceiverConfig::carrier_reference_cycles_per_sample).def_readwrite("phase_reference_radians", &ReceiverConfig::phase_reference_radians).def_readwrite("noise_variance", &ReceiverConfig::noise_variance).def_readwrite("pulse_shape", &ReceiverConfig::pulse_shape).def_readwrite("rrc_rolloff", &ReceiverConfig::rrc_rolloff).def_readwrite("acquisition_symbols", &ReceiverConfig::acquisition_symbols).def_readwrite("maximum_buffered_samples", &ReceiverConfig::maximum_buffered_samples);
    py::class_<ReceiverResult>(module, "NativeReceiverResult")
        .def_property_readonly("symbols", [](const ReceiverResult& value) { return vector_view(value.symbols); })
        .def_property_readonly("hard_bits", [](const ReceiverResult& value) { return vector_view(value.hard_bits); })
        .def_property_readonly("soft_llrs", [](const ReceiverResult& value) { return vector_view(value.soft_llrs); })
        .def_property_readonly("sample_offsets", [](const ReceiverResult& value) { return vector_view(value.sample_offsets); })
        .def_readonly("acquisition_status", &ReceiverResult::acquisition_status).def_readonly("mapping_status", &ReceiverResult::mapping_status).def_readonly("modulation", &ReceiverResult::modulation).def_readonly("bits_per_symbol", &ReceiverResult::bits_per_symbol).def_readonly("carrier_offset", &ReceiverResult::carrier_offset).def_readonly("carrier_offset_unit", &ReceiverResult::carrier_offset_unit).def_readonly("timing_offset_samples", &ReceiverResult::timing_offset_samples).def_readonly("timing_error", &ReceiverResult::timing_error).def_readonly("carrier_error", &ReceiverResult::carrier_error).def_readonly("evm_percent", &ReceiverResult::evm_percent).def_readonly("noise_variance", &ReceiverResult::noise_variance).def_readonly("unresolved_phase_rotations", &ReceiverResult::unresolved_phase_rotations).def_readonly("unresolved_carrier_offsets", &ReceiverResult::unresolved_carrier_offsets).def_readonly("diagnostics", &ReceiverResult::diagnostics);
    py::class_<ReceiverSession>(module, "ReceiverSession").def(py::init<ReceiverConfig>())
        .def("process", [](ReceiverSession& session, const py::array& input) { const auto samples = complex_span(input); py::gil_scoped_release release; return session.process(samples); })
        .def("flush", [](ReceiverSession& session) { py::gil_scoped_release release; return session.flush(); })
        .def("reset", &ReceiverSession::reset).def_property_readonly("buffered_samples", &ReceiverSession::buffered_samples);
    module.def("demodulate", [](const py::array& input, const ReceiverConfig& config) { const auto samples = complex_span(input); py::gil_scoped_release release; return demodulate(samples, config); }, py::arg("samples"), py::arg("config"));
}
