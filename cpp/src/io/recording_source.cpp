#include "sih/io/recording_source.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace sih::io {
namespace {

[[nodiscard]] std::uint16_t read_u16_le(const std::byte* data) {
    return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(data[0])) |
           (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(data[1])) << 8U);
}

[[nodiscard]] std::uint32_t read_u32_le(const std::byte* data) {
    return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[0])) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[1])) << 8U) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[2])) << 16U) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[3])) << 24U);
}

[[nodiscard]] std::uint32_t read_u32(const std::byte* data, const ByteOrder order) {
    if (order == ByteOrder::little) {
        return read_u32_le(data);
    }
    return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[3])) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[2])) << 8U) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[1])) << 16U) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[0])) << 24U);
}

[[nodiscard]] std::int64_t signed_value(
    const std::byte* data, const std::size_t bytes, const ByteOrder order) {
    std::uint64_t raw = 0;
    for (std::size_t index = 0; index < bytes; ++index) {
        const auto source = order == ByteOrder::little ? index : bytes - 1U - index;
        raw |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(data[source])) << (8U * index);
    }
    const auto bits = bytes * 8U;
    const auto sign = std::uint64_t{1} << (bits - 1U);
    if ((raw & sign) != 0U) {
        return static_cast<std::int64_t>(raw) -
               static_cast<std::int64_t>(std::uint64_t{1} << bits);
    }
    return static_cast<std::int64_t>(raw);
}

[[nodiscard]] float decode_scalar(
    const std::byte* data, const ScalarFormat format, const ByteOrder order) {
    switch (format) {
        case ScalarFormat::f32:
        case ScalarFormat::complex_f32: {
            const auto raw = read_u32(data, order);
            const float value = std::bit_cast<float>(raw);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("floating-point recording contains NaN or infinity");
            }
            return value;
        }
        case ScalarFormat::s8:
            return static_cast<float>(signed_value(data, 1, order)) / 128.0F;
        case ScalarFormat::u8:
            return (static_cast<float>(std::to_integer<std::uint8_t>(data[0])) - 128.0F) / 128.0F;
        case ScalarFormat::s16:
            return static_cast<float>(signed_value(data, 2, order)) / 32768.0F;
        case ScalarFormat::s24:
            return static_cast<float>(signed_value(data, 3, order)) / 8388608.0F;
        case ScalarFormat::s32:
            return static_cast<float>(signed_value(data, 4, order)) / 2147483648.0F;
    }
    throw std::invalid_argument("unsupported scalar format");
}

[[nodiscard]] std::size_t scalar_bytes(const ScalarFormat format) {
    switch (format) {
        case ScalarFormat::s8:
        case ScalarFormat::u8: return 1;
        case ScalarFormat::s16: return 2;
        case ScalarFormat::s24: return 3;
        case ScalarFormat::s32:
        case ScalarFormat::f32: return 4;
        case ScalarFormat::complex_f32: return 8;
    }
    throw std::invalid_argument("unsupported scalar format");
}

[[nodiscard]] std::vector<std::byte> read_exact(
    const std::filesystem::path& path,
    const std::uint64_t offset,
    const std::size_t size) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("unable to open recording: " + path.string());
    }
    stream.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> bytes(size);
    if (size > 0U && !stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("recording ended before the requested chunk was read");
    }
    return bytes;
}

[[nodiscard]] bool fourcc(const std::array<char, 4>& value, const std::string_view expected) {
    return std::equal(value.begin(), value.end(), expected.begin(), expected.end());
}

}  // namespace

RawIQSource::RawIQSource(
    std::filesystem::path path,
    const ScalarFormat format,
    const IQOrder iq_order,
    const ByteOrder byte_order,
    const std::optional<double> sample_rate_hz)
    : path_(std::move(path)), format_(format), iq_order_(iq_order), byte_order_(byte_order) {
    if (!std::filesystem::is_regular_file(path_)) {
        throw std::invalid_argument("raw IQ path is not a regular file");
    }
    if (sample_rate_hz && (!std::isfinite(*sample_rate_hz) || *sample_rate_hz <= 0.0)) {
        throw std::invalid_argument("sample rate must be finite and positive when supplied");
    }
    bytes_per_sample_ = format == ScalarFormat::complex_f32 ? 8U : scalar_bytes(format) * 2U;
    const auto size = std::filesystem::file_size(path_);
    if (size % bytes_per_sample_ != 0U) {
        throw std::invalid_argument("raw IQ file size is not a multiple of the configured sample size");
    }
    info_.sample_count = size / bytes_per_sample_;
    info_.channel_count = 2;
    info_.scalar_format = format;
    info_.sample_rate_hz = sample_rate_hz;
    info_.complex_samples = true;
    info_.normalized = format != ScalarFormat::f32 && format != ScalarFormat::complex_f32;
    if (format == ScalarFormat::u8) {
        info_.conversion = "unsigned 8-bit IQ centered at 128 and scaled by 1/128";
    } else if (info_.normalized) {
        info_.conversion = "signed integer IQ scaled by its negative full-scale magnitude";
    } else {
        info_.conversion = "floating-point IQ converted to complex64 without amplitude scaling";
    }
}

const SourceInfo& RawIQSource::info() const noexcept { return info_; }

SampleChunk RawIQSource::read_chunk(const std::uint64_t start, const std::size_t count) const {
    if (start > info_.sample_count) {
        throw std::out_of_range("chunk start exceeds source sample count");
    }
    const auto available = info_.sample_count - start;
    const auto actual = static_cast<std::size_t>(std::min<std::uint64_t>(available, count));
    const auto bytes = read_exact(path_, start * bytes_per_sample_, actual * bytes_per_sample_);
    SampleChunk result;
    result.start_sample = start;
    result.source_sample_count = info_.sample_count;
    result.samples->resize(actual);
    const auto component_bytes = format_ == ScalarFormat::complex_f32 ? 4U : scalar_bytes(format_);
    for (std::size_t index = 0; index < actual; ++index) {
        const auto* pair = bytes.data() + index * bytes_per_sample_;
        const float first = decode_scalar(pair, format_, byte_order_);
        const float second = decode_scalar(pair + component_bytes, format_, byte_order_);
        (*result.samples)[index] = iq_order_ == IQOrder::iq
            ? std::complex<float>{first, second}
            : std::complex<float>{second, first};
    }
    return result;
}

WavSource::WavSource(std::filesystem::path path, const WavChannelMode mode)
    : path_(std::move(path)), mode_(mode) {
    std::ifstream stream(path_, std::ios::binary);
    if (!stream) {
        throw std::invalid_argument("unable to open WAV file");
    }
    std::array<char, 12> riff{};
    if (!stream.read(riff.data(), static_cast<std::streamsize>(riff.size())) ||
        std::string_view(riff.data(), 4) != "RIFF" || std::string_view(riff.data() + 8, 4) != "WAVE") {
        throw std::invalid_argument("file is not a little-endian RIFF/WAVE recording");
    }

    bool found_format = false;
    bool found_data = false;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t block_align = 0;
    while (stream && !(found_format && found_data)) {
        std::array<char, 4> id{};
        std::array<std::byte, 4> size_bytes{};
        if (!stream.read(id.data(), 4) ||
            !stream.read(reinterpret_cast<char*>(size_bytes.data()), 4)) {
            break;
        }
        const auto chunk_size = read_u32_le(size_bytes.data());
        const auto payload = static_cast<std::uint64_t>(stream.tellg());
        if (fourcc(id, "fmt ")) {
            if (chunk_size < 16U) {
                throw std::invalid_argument("WAV fmt chunk is truncated");
            }
            const auto format = read_exact(path_, payload, std::min<std::uint32_t>(chunk_size, 40U));
            wav_format_ = read_u16_le(format.data());
            channels = read_u16_le(format.data() + 2);
            sample_rate = read_u32_le(format.data() + 4);
            block_align = read_u16_le(format.data() + 12);
            bits_per_sample_ = read_u16_le(format.data() + 14);
            if (wav_format_ == 0xFFFEU) {
                constexpr std::array<std::uint8_t, 12> guid_suffix{
                    0x00, 0x00, 0x10, 0x00, 0x80, 0x00,
                    0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
                if (format.size() < 40U || read_u16_le(format.data() + 16) < 22U ||
                    !std::equal(
                        guid_suffix.begin(), guid_suffix.end(),
                        reinterpret_cast<const std::uint8_t*>(format.data() + 28))) {
                    throw std::invalid_argument("unsupported WAVE_FORMAT_EXTENSIBLE subformat");
                }
                wav_format_ = read_u16_le(format.data() + 24);
            }
            found_format = true;
        } else if (fourcc(id, "data")) {
            data_offset_ = payload;
            data_size_ = chunk_size;
            found_data = true;
        }
        stream.seekg(static_cast<std::streamoff>(payload + chunk_size + (chunk_size & 1U)));
    }
    if (!found_format || !found_data) {
        throw std::invalid_argument("WAV requires both fmt and data chunks");
    }
    if ((wav_format_ != 1U && wav_format_ != 3U) || (channels != 1U && channels != 2U)) {
        throw std::invalid_argument("WAV must be PCM/IEEE-float with one or two channels");
    }
    if (sample_rate == 0U) {
        throw std::invalid_argument("WAV sample rate must be positive");
    }
    if ((wav_format_ == 1U && bits_per_sample_ != 8U && bits_per_sample_ != 16U &&
         bits_per_sample_ != 24U && bits_per_sample_ != 32U) ||
        (wav_format_ == 3U && bits_per_sample_ != 32U)) {
        throw std::invalid_argument("unsupported WAV sample representation");
    }
    if (block_align == 0U || data_size_ % block_align != 0U ||
        block_align != channels * (bits_per_sample_ / 8U)) {
        throw std::invalid_argument("WAV data size or block alignment is invalid");
    }
    if (data_offset_ + data_size_ > std::filesystem::file_size(path_)) {
        throw std::invalid_argument("WAV data chunk extends beyond the file");
    }
    if (channels == 1U && mode_ != WavChannelMode::mono && mode_ != WavChannelMode::channel_0) {
        throw std::invalid_argument("mono WAV only supports mono/channel_0 interpretation");
    }
    if (channels == 2U && mode_ == WavChannelMode::mono) {
        throw std::invalid_argument("stereo WAV requires explicit channel_0, channel_1, or stereo_iq interpretation");
    }
    bytes_per_frame_ = block_align;
    info_.sample_count = data_size_ / bytes_per_frame_;
    info_.channel_count = channels;
    info_.sample_rate_hz = static_cast<double>(sample_rate);
    info_.complex_samples = mode_ == WavChannelMode::stereo_iq;
    info_.normalized = wav_format_ == 1U;
    if (wav_format_ == 3U) {
        info_.scalar_format = ScalarFormat::f32;
    } else if (bits_per_sample_ == 8U) {
        info_.scalar_format = ScalarFormat::u8;
    } else if (bits_per_sample_ == 16U) {
        info_.scalar_format = ScalarFormat::s16;
    } else if (bits_per_sample_ == 24U) {
        info_.scalar_format = ScalarFormat::s24;
    } else {
        info_.scalar_format = ScalarFormat::s32;
    }
    info_.conversion = wav_format_ == 1U
        ? "PCM WAV converted to normalized complex64; unsigned PCM8 centered at 128"
        : "IEEE-float WAV converted to complex64 without amplitude scaling";
}

const SourceInfo& WavSource::info() const noexcept { return info_; }

SampleChunk WavSource::read_chunk(const std::uint64_t start, const std::size_t count) const {
    if (start > info_.sample_count) {
        throw std::out_of_range("chunk start exceeds source sample count");
    }
    const auto available = info_.sample_count - start;
    const auto actual = static_cast<std::size_t>(std::min<std::uint64_t>(available, count));
    const auto bytes = read_exact(path_, data_offset_ + start * bytes_per_frame_, actual * bytes_per_frame_);
    SampleChunk result;
    result.start_sample = start;
    result.source_sample_count = info_.sample_count;
    result.samples->resize(actual);
    const auto component_bytes = bits_per_sample_ / 8U;
    const auto format = wav_format_ == 3U ? ScalarFormat::f32 : info_.scalar_format;
    for (std::size_t index = 0; index < actual; ++index) {
        const auto* frame = bytes.data() + index * bytes_per_frame_;
        const float first = decode_scalar(frame, format, ByteOrder::little);
        float second = 0.0F;
        if (info_.channel_count == 2U) {
            second = decode_scalar(frame + component_bytes, format, ByteOrder::little);
        }
        if (mode_ == WavChannelMode::stereo_iq) {
            (*result.samples)[index] = {first, second};
        } else if (mode_ == WavChannelMode::channel_1) {
            (*result.samples)[index] = {second, 0.0F};
        } else {
            (*result.samples)[index] = {first, 0.0F};
        }
    }
    return result;
}

}  // namespace sih::io
