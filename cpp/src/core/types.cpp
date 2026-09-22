#include "sih/core/types.hpp"

namespace sih::core {

void CancellationToken::cancel() noexcept {
    cancelled_.store(true, std::memory_order_release);
}

bool CancellationToken::is_cancelled() const noexcept {
    return cancelled_.load(std::memory_order_acquire);
}

void ProgressState::update(const std::uint64_t completed, const std::uint64_t total) noexcept {
    total_.store(total, std::memory_order_relaxed);
    completed_.store(completed, std::memory_order_release);
}

std::pair<std::uint64_t, std::uint64_t> ProgressState::snapshot() const noexcept {
    const auto completed = completed_.load(std::memory_order_acquire);
    const auto total = total_.load(std::memory_order_relaxed);
    return {completed, total};
}

double ProgressState::fraction() const noexcept {
    const auto [completed, total] = snapshot();
    if (total == 0) {
        return 0.0;
    }
    return static_cast<double>(completed) / static_cast<double>(total);
}

const char* to_string(const Severity value) noexcept {
    switch (value) {
        case Severity::info: return "INFO";
        case Severity::warning: return "WARNING";
        case Severity::error: return "ERROR";
    }
    return "ERROR";
}

const char* to_string(const ExecutionStatus value) noexcept {
    switch (value) {
        case ExecutionStatus::completed: return "COMPLETED";
        case ExecutionStatus::cancelled: return "CANCELLED";
    }
    return "CANCELLED";
}

const char* to_string(const EvidenceStatus value) noexcept {
    switch (value) {
        case EvidenceStatus::unverified: return "UNVERIFIED";
        case EvidenceStatus::verified: return "VERIFIED";
        case EvidenceStatus::rejected: return "REJECTED";
    }
    return "REJECTED";
}

}  // namespace sih::core
