#include "sih/core/types.hpp"
#include "sih/fec/viterbi.hpp"
#include "phase2_bindings.hpp"
#include "phase3_bindings.hpp"
#include "phase4_bindings.hpp"
#include "phase5_bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <span>
#include <stdexcept>

namespace py = pybind11;

namespace {

py::array decoded_bits_view(const sih::fec::ViterbiResult& result) {
    auto* owner = new std::shared_ptr<std::vector<std::uint8_t>>(result.decoded_bits);
    py::capsule capsule(owner, [](void* pointer) {
        delete static_cast<std::shared_ptr<std::vector<std::uint8_t>>*>(pointer);
    });
    return py::array(
        py::dtype::of<std::uint8_t>(),
        {static_cast<py::ssize_t>((*owner)->size())},
        {static_cast<py::ssize_t>(sizeof(std::uint8_t))},
        (*owner)->data(),
        capsule);
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "SIH26147 native compute core";
    module.attr("__version__") = SIH_NATIVE_VERSION;
    module.attr("API_VERSION") = 5;

    py::enum_<sih::core::Severity>(module, "Severity")
        .value("INFO", sih::core::Severity::info)
        .value("WARNING", sih::core::Severity::warning)
        .value("ERROR", sih::core::Severity::error);
    py::enum_<sih::core::ExecutionStatus>(module, "ExecutionStatus")
        .value("COMPLETED", sih::core::ExecutionStatus::completed)
        .value("CANCELLED", sih::core::ExecutionStatus::cancelled);
    py::enum_<sih::core::EvidenceStatus>(module, "EvidenceStatus")
        .value("UNVERIFIED", sih::core::EvidenceStatus::unverified)
        .value("VERIFIED", sih::core::EvidenceStatus::verified)
        .value("REJECTED", sih::core::EvidenceStatus::rejected);

    py::class_<sih::core::Diagnostic>(module, "Diagnostic")
        .def(py::init<>())
        .def_readwrite("severity", &sih::core::Diagnostic::severity)
        .def_readwrite("code", &sih::core::Diagnostic::code)
        .def_readwrite("message", &sih::core::Diagnostic::message)
        .def_readwrite("evidence", &sih::core::Diagnostic::evidence);

    py::class_<sih::core::CancellationToken, std::shared_ptr<sih::core::CancellationToken>>(
        module, "CancellationToken")
        .def(py::init<>())
        .def("cancel", &sih::core::CancellationToken::cancel)
        .def_property_readonly("is_cancelled", &sih::core::CancellationToken::is_cancelled);

    py::class_<sih::core::ProgressState, std::shared_ptr<sih::core::ProgressState>>(
        module, "ProgressState")
        .def(py::init<>())
        .def("update", &sih::core::ProgressState::update)
        .def_property_readonly("snapshot", &sih::core::ProgressState::snapshot)
        .def_property_readonly("fraction", &sih::core::ProgressState::fraction);

    py::class_<sih::fec::ViterbiResult>(module, "ViterbiResult")
        .def_property_readonly("decoded_bits", &decoded_bits_view)
        .def_readonly("execution_status", &sih::fec::ViterbiResult::execution_status)
        .def_readonly("decode_validity", &sih::fec::ViterbiResult::decode_validity)
        .def_readonly("path_metric_margin", &sih::fec::ViterbiResult::path_metric_margin)
        .def_readonly("consumed_llrs", &sih::fec::ViterbiResult::consumed_llrs)
        .def_readonly("residual_llrs", &sih::fec::ViterbiResult::residual_llrs);

    module.def(
        "decode_viterbi_k7_r12",
        [](const py::array_t<float, py::array::c_style>& llrs,
           const std::shared_ptr<sih::core::CancellationToken>& cancellation,
           const std::shared_ptr<sih::core::ProgressState>& progress) {
            if (llrs.ndim() != 1) {
                throw std::invalid_argument("Viterbi input must be a one-dimensional float32 array");
            }
            const auto* data = llrs.data();
            const auto size = static_cast<std::size_t>(llrs.size());
            py::gil_scoped_release release;
            return sih::fec::decode_k7_r12_soft(
                std::span<const float>(data, size), cancellation, progress);
        },
        py::arg("llrs"),
        py::arg("cancellation") = nullptr,
        py::arg("progress") = nullptr,
        "Decode K=7, rate-1/2 (171,133 octal) soft LLRs; positive LLR means bit 1.");

    bind_phase2(module);
    bind_phase3(module);
    bind_phase4(module);
    bind_phase5(module);
}
