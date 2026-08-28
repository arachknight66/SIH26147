from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts.generate_digital_dataset import generate_digital_stream

def generate_demo_dataset(output_dir: str = "examples") -> dict[str, str]:
    """
    Generate demonstration dataset with isolated ground-truth manifests.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    def _save_raw_iq(filename: str, samples: np.ndarray, truth_info: dict) -> Path:
        iq_path = out / filename
        samples.astype(np.complex64).tofile(iq_path)
        manifest_path = out / f"{filename}.manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(truth_info, f, indent=2)
        generated[filename] = str(iq_path.resolve())
        return iq_path

    # 1. Clean QPSK (Protocol A)
    rx_a, _, _ = generate_digital_stream(protocol="PROTOCOL_A", num_frames=5, seed=42)
    syms_a = ((np.where(rx_a[0::2] == 0, 1.0, -1.0) + 1j * np.where(rx_a[1::2] == 0, 1.0, -1.0)) / np.sqrt(2.0)).astype(np.complex64)
    samples_a = np.repeat(syms_a, 4) # 4 SPS
    _save_raw_iq("clean_qpsk.iq", samples_a, {"ground_truth_modulation": "QPSK", "fec": "NONE", "expected_status": "INDEPENDENTLY_VERIFIED"})

    # 2. Noisy QPSK (Protocol C with Convolutional FEC)
    rx_c, _, _ = generate_digital_stream(protocol="PROTOCOL_C", num_frames=5, ber=0.005, seed=42)
    syms_c = ((np.where(rx_c[0::2] == 0, 1.0, -1.0) + 1j * np.where(rx_c[1::2] == 0, 1.0, -1.0)) / np.sqrt(2.0)).astype(np.complex64)
    noise_c = (np.random.normal(0, 0.15, len(syms_c)) + 1j * np.random.normal(0, 0.15, len(syms_c))).astype(np.complex64)
    samples_c = np.repeat(syms_c + noise_c, 4)
    _save_raw_iq("noisy_qpsk_fec.iq", samples_c, {"ground_truth_modulation": "QPSK", "fec": "CONV_K7_R12_NASA", "expected_status": "INDEPENDENTLY_VERIFIED"})

    # 3. Scrambled QPSK (Protocol D)
    rx_d, _, _ = generate_digital_stream(protocol="PROTOCOL_D", num_frames=5, seed=42)
    syms_d = ((np.where(rx_d[0::2] == 0, 1.0, -1.0) + 1j * np.where(rx_d[1::2] == 0, 1.0, -1.0)) / np.sqrt(2.0)).astype(np.complex64)
    samples_d = np.repeat(syms_d, 4)
    _save_raw_iq("scrambled_frame.iq", samples_d, {"ground_truth_modulation": "QPSK", "scrambler": "ITU_V29", "expected_status": "INDEPENDENTLY_VERIFIED"})

    # 4. Pure Noise (Out-of-Distribution)
    noise_samples = (np.random.normal(0, 1.0, 4096) + 1j * np.random.normal(0, 1.0, 4096)).astype(np.complex64)
    _save_raw_iq("pure_noise.iq", noise_samples, {"ground_truth": "NOISE", "expected_status": "REJECTED_OR_INSUFFICIENT_EVIDENCE"})

    # 5. Adversarial Random Stream
    adv_bits = np.random.randint(0, 2, 4096, dtype=np.uint8)
    adv_syms = ((np.where(adv_bits[0::2] == 0, 1.0, -1.0) + 1j * np.where(adv_bits[1::2] == 0, 1.0, -1.0)) / np.sqrt(2.0)).astype(np.complex64)
    adv_samples = np.repeat(adv_syms, 4)
    _save_raw_iq("adversarial_random.iq", adv_samples, {"ground_truth": "ADVERSARIAL_RANDOM", "expected_status": "REJECTED_OR_FALSIFIED"})

    return generated

if __name__ == "__main__":
    files = generate_demo_dataset()
    print("Demonstration dataset created in examples/:")
    for k, v in files.items():
        print(f"  • {k}: {v}")
