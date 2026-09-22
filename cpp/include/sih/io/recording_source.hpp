#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sih::io {

enum class ScalarFormat : std::uint8_t {
    complex_f32,
    f32,
    s8,
    u8,
    s16,
    s24,
    s32,
};

enum class ByteOrder : std::uint8_t { little, big };
enum class IQOrder : std::uint8_t { iq, qi };
enum class WavChannelMode : std::uint8_t { mono, channel_0, channel_1, stereo_iq };

struct SourceInfo {
    std::uint64_t sample_count{0};
    std::uint32_t channel_count{0};
    ScalarFormat scalar_format{ScalarFormat::f32};
    std::optional<double> sample_rate_hz;
    bool complex_samples{false};
    bool normalized{true};
    std::string conversion;
};

struct SampleChunk {
    std::shared_ptr<std::vector<std::complex<float>>> samples{
        std::make_shared<std::vector<std::complex<float>>>()};
    std::uint64_t start_sample{0};
    std::uint64_t source_sample_count{0};
};

class RawIQSource final {
public:
    RawIQSource(
        std::filesystem::path path,
        ScalarFormat format,
        IQOrder iq_order,
        ByteOrder byte_order,
        std::optional<double> sample_rate_hz = {});

    [[nodiscard]] const SourceInfo& info() const noexcept;
    [[nodiscard]] SampleChunk read_chunk(std::uint64_t start, std::size_t count) const;

private:
    std::filesystem::path path_;
    ScalarFormat format_;
    IQOrder iq_order_;
    ByteOrder byte_order_;
    SourceInfo info_;
    std::size_t bytes_per_sample_{0};
};

class WavSource final {
public:
    WavSource(std::filesystem::path path, WavChannelMode mode);

    [[nodiscard]] const SourceInfo& info() const noexcept;
    [[nodiscard]] SampleChunk read_chunk(std::uint64_t start, std::size_t count) const;

private:
    std::filesystem::path path_;
    WavChannelMode mode_;
    SourceInfo info_;
    std::uint16_t wav_format_{0};
    std::uint16_t bits_per_sample_{0};
    std::uint64_t data_offset_{0};
    std::uint64_t data_size_{0};
    std::size_t bytes_per_frame_{0};
};

}  // namespace sih::io
