#include "phase5_bindings.hpp"
#include "sih/bitstream/bitstream.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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

template <typename Value>
std::span<const Value> array_span(const py::array& input, const char* name) {
    if (!input.dtype().is(py::dtype::of<Value>())) throw py::type_error(std::string(name) + " has an invalid dtype; implicit conversion is disabled");
    if (input.ndim() != 1) throw std::invalid_argument(std::string(name) + " must be one-dimensional");
    if ((input.flags() & py::array::c_style) == 0) throw std::invalid_argument(std::string(name) + " must be C-contiguous; implicit copies are disabled");
    return {static_cast<const Value*>(input.data()), static_cast<std::size_t>(input.size())};
}

}

void bind_phase5(py::module_& module) {
    using namespace sih::bitstream;
    py::enum_<InterleaverFamily>(module, "InterleaverFamily")
        .value("NONE", InterleaverFamily::none).value("BLOCK", InterleaverFamily::block)
        .value("CONVOLUTIONAL", InterleaverFamily::convolutional).value("DIAGONAL", InterleaverFamily::diagonal)
        .value("PSEUDO_RANDOM", InterleaverFamily::pseudo_random);
    py::class_<InterleaverConfig>(module, "NativeInterleaverConfig")
        .def(py::init<>()).def_readwrite("family", &InterleaverConfig::family).def_readwrite("rows", &InterleaverConfig::rows)
        .def_readwrite("columns", &InterleaverConfig::columns).def_readwrite("read_by_row", &InterleaverConfig::read_by_row)
        .def_readwrite("branches", &InterleaverConfig::branches).def_readwrite("delay", &InterleaverConfig::delay)
        .def_readwrite("seed", &InterleaverConfig::seed).def_readwrite("permutation", &InterleaverConfig::permutation);
    py::class_<DeinterleaveResult>(module, "NativeDeinterleaveResult")
        .def_property_readonly("bits", [](const DeinterleaveResult& result) { return vector_view(result.bits); })
        .def_property_readonly("llrs", [](const DeinterleaveResult& result) { return vector_view(result.llrs); })
        .def_readonly("transformed_bits", &DeinterleaveResult::transformed_bits).def_readonly("residual_bits", &DeinterleaveResult::residual_bits)
        .def_readonly("diagnostics", &DeinterleaveResult::diagnostics);
    py::class_<CorrelationPattern>(module, "CorrelationPattern")
        .def(py::init<>()).def_readwrite("name", &CorrelationPattern::name).def_readwrite("bits", &CorrelationPattern::bits);
    py::class_<CorrelationMatch>(module, "NativeCorrelationMatch")
        .def_readonly("pattern_name", &CorrelationMatch::pattern_name).def_readonly("bit_offset", &CorrelationMatch::bit_offset)
        .def_readonly("hamming_distance", &CorrelationMatch::hamming_distance).def_readonly("confidence", &CorrelationMatch::confidence)
        .def_readonly("periodic", &CorrelationMatch::periodic);
    py::class_<CrcConfig>(module, "CrcConfig")
        .def(py::init<>()).def_readwrite("name", &CrcConfig::name).def_readwrite("width", &CrcConfig::width)
        .def_readwrite("polynomial", &CrcConfig::polynomial).def_readwrite("initial", &CrcConfig::initial)
        .def_readwrite("reflect_input", &CrcConfig::reflect_input).def_readwrite("reflect_output", &CrcConfig::reflect_output)
        .def_readwrite("xor_output", &CrcConfig::xor_output);
    py::class_<ReedSolomonConfig>(module, "ReedSolomonConfig")
        .def(py::init<>()).def_readwrite("n", &ReedSolomonConfig::n).def_readwrite("k", &ReedSolomonConfig::k);
    py::class_<ReedSolomonResult>(module, "NativeReedSolomonResult")
        .def_property_readonly("decoded_bytes", [](const ReedSolomonResult& result) { return vector_view(result.decoded_bytes); })
        .def_readonly("success", &ReedSolomonResult::success).def_readonly("corrected_symbols", &ReedSolomonResult::corrected_symbols)
        .def_readonly("consumed_bytes", &ReedSolomonResult::consumed_bytes).def_readonly("residual_bytes", &ReedSolomonResult::residual_bytes)
        .def_readonly("syndrome_weight", &ReedSolomonResult::syndrome_weight).def_readonly("diagnostics", &ReedSolomonResult::diagnostics);
    py::class_<LdpcMatrix>(module, "LdpcMatrix")
        .def(py::init<>()).def_readwrite("variable_count", &LdpcMatrix::variable_count).def_readwrite("checks", &LdpcMatrix::checks)
        .def_static("from_alist", &load_alist);
    py::class_<LdpcConfig>(module, "LdpcConfig")
        .def(py::init<>()).def_readwrite("maximum_iterations", &LdpcConfig::maximum_iterations).def_readwrite("normalization", &LdpcConfig::normalization);
    py::class_<LdpcResult>(module, "NativeLdpcResult")
        .def_property_readonly("decoded_bits", [](const LdpcResult& result) { return vector_view(result.decoded_bits); })
        .def_readonly("converged", &LdpcResult::converged).def_readonly("iterations", &LdpcResult::iterations)
        .def_readonly("syndrome_weight", &LdpcResult::syndrome_weight).def_readonly("diagnostics", &LdpcResult::diagnostics);

    module.def("deinterleave", [](const py::array& bits, const py::array& llrs, const InterleaverConfig& config) {
        const auto bit_values = array_span<std::uint8_t>(bits, "bits"); const auto llr_values = array_span<float>(llrs, "llrs"); py::gil_scoped_release release; return deinterleave(bit_values, llr_values, config);
    });
    module.def("interleave_bits", [](const py::array& bits, const InterleaverConfig& config) {
        const auto bit_values = array_span<std::uint8_t>(bits, "bits"); py::gil_scoped_release release; return interleave_bits(bit_values, config);
    });
    module.def("interleave_llrs", [](const py::array& llrs, const InterleaverConfig& config) {
        const auto values = array_span<float>(llrs, "llrs"); py::gil_scoped_release release; return interleave_llrs(values, config);
    });
    module.def("correlate_bits", [](const py::array& bits, const py::array& llrs, const std::vector<CorrelationPattern>& patterns, const float maximum_fraction) {
        const auto bit_values = array_span<std::uint8_t>(bits, "bits"); const auto llr_values = array_span<float>(llrs, "llrs"); py::gil_scoped_release release; return correlate(bit_values, llr_values, patterns, maximum_fraction);
    }, py::arg("bits"), py::arg("llrs"), py::arg("patterns"), py::arg("maximum_hamming_fraction") = 0.15F);
    module.def("crc_bits", [](const py::array& bits, const CrcConfig& config) { const auto values = array_span<std::uint8_t>(bits, "bits"); py::gil_scoped_release release; return crc_bits(values, config); });
    module.def("encode_reed_solomon", [](const py::array& message, const ReedSolomonConfig& config) { const auto values = array_span<std::uint8_t>(message, "message"); py::gil_scoped_release release; return encode_reed_solomon(values, config); });
    module.def("decode_reed_solomon", [](const py::array& bytes, const ReedSolomonConfig& config) { const auto values = array_span<std::uint8_t>(bytes, "bytes"); py::gil_scoped_release release; return decode_reed_solomon(values, config); });
    module.def("decode_ldpc", [](const py::array& llrs, const LdpcMatrix& matrix, const LdpcConfig& config) { const auto values = array_span<float>(llrs, "llrs"); py::gil_scoped_release release; return decode_ldpc_normalized_min_sum(values, matrix, config); });
}
