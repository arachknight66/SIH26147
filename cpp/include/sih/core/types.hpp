#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>

namespace sih::core {

enum class Severity : std::uint8_t {
    info,
    warning,
    error,
};

enum class ExecutionStatus : std::uint8_t {
    completed,
    cancelled,
};

enum class EvidenceStatus : std::uint8_t {
    unverified,
    verified,
    rejected,
};

struct Diagnostic {
    Severity severity{Severity::info};
    std::string code;
    std::string message;
    std::string evidence;
};

class CancellationToken final {
public:
    void cancel() noexcept;
    [[nodiscard]] bool is_cancelled() const noexcept;

private:
    std::atomic_bool cancelled_{false};
};

class ProgressState final {
public:
    void update(std::uint64_t completed, std::uint64_t total) noexcept;
    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> snapshot() const noexcept;
    [[nodiscard]] double fraction() const noexcept;

private:
    std::atomic<std::uint64_t> completed_{0};
    std::atomic<std::uint64_t> total_{0};
};

[[nodiscard]] const char* to_string(Severity value) noexcept;
[[nodiscard]] const char* to_string(ExecutionStatus value) noexcept;
[[nodiscard]] const char* to_string(EvidenceStatus value) noexcept;

}  // namespace sih::core
