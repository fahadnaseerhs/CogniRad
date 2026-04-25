"""
CogniRad - Task 1.4: RadioML -> 3-Class Remap

Classes:
  BUSY   -> all real RadioML modulations at sufficiently high SNR
  FREE   -> synthetic low-energy background noise with varying noise floor
  JAMMED -> synthetic interference from multiple jammer families

The output schema is intentionally unchanged:
  X   : IQ samples
  y   : integer labels (BUSY=0, FREE=1, JAMMED=2)
  snr : float metadata column
"""

import gc
import os
import sys
import time

import h5py
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================
COLAB = False

if COLAB:
    FILE_PATH = "/content/drive/MyDrive/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
    OUTPUT_PATH = "/content/drive/MyDrive/dataset/radioml_remapped.hdf5"
else:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # go up from layer_1/ to CogniRad/
    FILE_PATH = os.path.join(_PROJECT_DIR, "dataset", "GOLD_XYZ_OSC.0001_1024.hdf5")
    OUTPUT_PATH = os.path.join(_PROJECT_DIR, "dataset", "radioml_remapped.hdf5")

SEED = 42
BUSY_SNR_MIN = 6.0
FREE_SIGMA_MIN = 0.10
FREE_SIGMA_MAX = 0.50
JAMMED_SIGMA_MIN = 2.00
JAMMED_SIGMA_MAX = 4.00

READ_BATCH = 10_000
SYNTH_BATCH = 4_096

LABEL_NAMES = {0: "BUSY", 1: "FREE", 2: "JAMMED"}
JAMMER_TYPES = ["wideband_noise", "chirp", "tone", "burst"]

CLASSES = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]


# ============================================================================
# HELPERS
# ============================================================================
def banner(text, width=70):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def section(text):
    print(f"\n-- {text}")


def progress_bar(done, total, bar_width=30):
    pct = done / max(total, 1)
    filled = int(bar_width * pct)
    bar = "#" * filled + "." * (bar_width - filled)
    return f"[{bar}] {done:>9,}/{total:,} ({pct * 100:5.1f}%)"


def sample_sigmas(n, sigma_min, sigma_max):
    return np.random.uniform(sigma_min, sigma_max, size=n).astype(np.float32)


def generate_free_batch(batch_size, seq_len, iq_dim):
    sigmas = sample_sigmas(batch_size, FREE_SIGMA_MIN, FREE_SIGMA_MAX)
    noise = np.random.normal(0.0, 1.0, size=(batch_size, seq_len, iq_dim)).astype(np.float32)
    noise *= sigmas[:, None, None]
    fake_snr = np.full(batch_size, -20.0, dtype=np.float32)
    return noise, fake_snr, sigmas


def generate_wideband_noise(count, seq_len):
    sigmas = sample_sigmas(count, JAMMED_SIGMA_MIN, JAMMED_SIGMA_MAX)
    x = np.random.normal(0.0, 1.0, size=(count, seq_len, 2)).astype(np.float32)
    x *= sigmas[:, None, None]
    return x, sigmas


def generate_chirp_jammers(count, seq_len):
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    x = np.zeros((count, seq_len, 2), dtype=np.float32)
    sigmas = sample_sigmas(count, JAMMED_SIGMA_MIN, JAMMED_SIGMA_MAX)

    for i in range(count):
        f_start = np.random.uniform(0.05, 0.20)
        f_end = np.random.uniform(0.30, 0.50)
        freq = f_start + (f_end - f_start) * t
        phase = 2.0 * np.pi * np.cumsum(freq).astype(np.float32)
        x[i, :, 0] = sigmas[i] * np.cos(phase)
        x[i, :, 1] = sigmas[i] * np.sin(phase)

    return x, sigmas


def generate_tone_jammers(count, seq_len):
    t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    x = np.zeros((count, seq_len, 2), dtype=np.float32)
    sigmas = sample_sigmas(count, JAMMED_SIGMA_MIN, JAMMED_SIGMA_MAX)

    for i in range(count):
        f_tone = np.random.uniform(0.10, 0.40)
        phase = 2.0 * np.pi * f_tone * t
        x[i, :, 0] = sigmas[i] * np.cos(phase)
        x[i, :, 1] = sigmas[i] * np.sin(phase)

    return x, sigmas


def generate_burst_jammers(count, seq_len):
    x = np.zeros((count, seq_len, 2), dtype=np.float32)
    sigmas = sample_sigmas(count, JAMMED_SIGMA_MIN, JAMMED_SIGMA_MAX)

    for i in range(count):
        n_bursts = np.random.randint(3, 8)
        for _ in range(n_bursts):
            start = np.random.randint(0, seq_len - 100)
            width = np.random.randint(30, 100)
            end = min(start + width, seq_len)
            burst = np.random.normal(0.0, sigmas[i], size=(end - start, 2)).astype(np.float32)
            x[i, start:end, :] = burst

    return x, sigmas


def split_counts(total, parts):
    counts = [total // parts] * parts
    for i in range(total % parts):
        counts[i] += 1
    return counts


def generate_jammed_batch(batch_size, seq_len, iq_dim):
    if iq_dim != 2:
        raise ValueError(f"Expected IQ dimension of 2, got {iq_dim}")

    counts = split_counts(batch_size, len(JAMMER_TYPES))
    generators = [
        generate_wideband_noise,
        generate_chirp_jammers,
        generate_tone_jammers,
        generate_burst_jammers,
    ]

    x_parts = []
    sigma_parts = []
    jammer_name_parts = []

    for count, jammer_name, generator in zip(counts, JAMMER_TYPES, generators):
        if count == 0:
            continue
        x_part, sigma_part = generator(count, seq_len)
        x_parts.append(x_part)
        sigma_parts.append(sigma_part)
        jammer_name_parts.extend([jammer_name] * count)

    x = np.concatenate(x_parts, axis=0)
    sigmas = np.concatenate(sigma_parts, axis=0)
    fake_snr = np.full(batch_size, 20.0, dtype=np.float32)

    shuffle_idx = np.random.permutation(batch_size)
    x = x[shuffle_idx]
    sigmas = sigmas[shuffle_idx]
    jammer_name_parts = [jammer_name_parts[i] for i in shuffle_idx]

    return x.astype(np.float32), fake_snr, sigmas, jammer_name_parts


def write_block(x_out, y_out, snr_out, x_block, y_value, snr_block):
    current = x_out.shape[0]
    slots = x_block.shape[0]
    x_out.resize(current + slots, axis=0)
    y_out.resize(current + slots, axis=0)
    snr_out.resize(current + slots, axis=0)
    x_out[current : current + slots] = x_block
    y_out[current : current + slots] = np.full(slots, y_value, dtype=np.int8)
    snr_out[current : current + slots] = snr_block.astype(np.float32)


def count_busy_samples(file_path, total_samples):
    """First pass: count BUSY candidates without loading full metadata into RAM."""
    busy_count = 0
    with h5py.File(file_path, "r") as f:
        for batch_start in range(0, total_samples, READ_BATCH):
            batch_end = min(batch_start + READ_BATCH, total_samples)
            z_batch = np.asarray(f["Z"][batch_start:batch_end], dtype=np.float32)
            snr_batch = z_batch[:, 0]
            busy_count += int((snr_batch >= BUSY_SNR_MIN).sum())
            del z_batch, snr_batch
    return busy_count


def stream_output_stats(file_path, sample_probe):
    """Low-memory verification pass over the remapped output file."""
    class_counts = {0: 0, 1: 0, 2: 0}
    snr_min = {0: np.inf, 1: np.inf, 2: np.inf}
    snr_max = {0: -np.inf, 1: -np.inf, 2: -np.inf}
    snr_sum = {0: 0.0, 1: 0.0, 2: 0.0}
    energy_sums = {0: 0.0, 1: 0.0, 2: 0.0}
    energy_counts = {0: 0, 1: 0, 2: 0}

    with h5py.File(file_path, "r") as f:
        total = int(f["y"].shape[0])
        seq_len = int(f["X"].shape[1])
        iq_dim = int(f["X"].shape[2])

        for batch_start in range(0, total, READ_BATCH):
            batch_end = min(batch_start + READ_BATCH, total)
            y_batch = np.asarray(f["y"][batch_start:batch_end], dtype=np.int8)
            snr_batch = np.asarray(f["snr"][batch_start:batch_end], dtype=np.float32)
            x_batch = np.asarray(f["X"][batch_start:batch_end], dtype=np.float32)

            for label in (0, 1, 2):
                mask = y_batch == label
                count = int(mask.sum())
                if count == 0:
                    continue

                class_counts[label] += count
                s = snr_batch[mask]
                snr_min[label] = min(snr_min[label], float(s.min()))
                snr_max[label] = max(snr_max[label], float(s.max()))
                snr_sum[label] += float(s.sum())

                remaining_probe = max(0, sample_probe - energy_counts[label])
                if remaining_probe > 0:
                    x_probe = x_batch[mask][:remaining_probe]
                    i_ch = x_probe[:, :, 0] if iq_dim == 2 else x_probe[:, 0, :]
                    q_ch = x_probe[:, :, 1] if iq_dim == 2 else x_probe[:, 1, :]
                    energy_sums[label] += float(np.sum(i_ch**2 + q_ch**2))
                    energy_counts[label] += int(x_probe.shape[0] * seq_len)

            del y_batch, snr_batch, x_batch

    energies = {}
    snr_means = {}
    for label in (0, 1, 2):
        energies[label] = (
            energy_sums[label] / energy_counts[label] if energy_counts[label] > 0 else 0.0
        )
        snr_means[label] = snr_sum[label] / class_counts[label] if class_counts[label] > 0 else 0.0

    return {
        "total": sum(class_counts.values()),
        "class_counts": class_counts,
        "snr_min": snr_min,
        "snr_max": snr_max,
        "snr_mean": snr_means,
        "energies": energies,
    }


# ============================================================================
# MAIN SCRIPT
# ============================================================================
np.random.seed(SEED)
banner("CogniRad - Task 1.4: RadioML -> 3-Class Remap")

if not os.path.exists(FILE_PATH):
    print(f"\n[ERROR] Input file not found:\n  {FILE_PATH}")
    sys.exit(1)

print(f"\nInput  : {FILE_PATH}")
print(f"Output : {OUTPUT_PATH}")

section("Step 1 - Clearing old output")
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)
    print("  Deleted previous output file.")
else:
    print("  No previous output found.")

section("Step 2 - Loading labels and SNR metadata")
print("  Metadata is streamed in batches to keep RAM usage low.")

with h5py.File(FILE_PATH, "r") as f:
    x_shape = f["X"].shape

total_samples = int(x_shape[0])
seq_len = int(x_shape[1])
iq_dim = int(x_shape[2])

print(f"  Total samples : {total_samples:,}")
print(f"  X shape       : {x_shape}")

section("Step 3 - Building BUSY mask")
busy_count = count_busy_samples(FILE_PATH, total_samples)
balance_target = busy_count

print(f"  BUSY definition : all {len(CLASSES)} RadioML modulations with SNR >= {BUSY_SNR_MIN:.1f} dB")
print(f"  BUSY candidates : {busy_count:,}")
print(f"  Balance target  : {balance_target:,} samples per class")
print(f"  Total output    : {balance_target * 3:,} samples")
print(f"  FREE sigma      : uniform[{FREE_SIGMA_MIN:.2f}, {FREE_SIGMA_MAX:.2f}]")
print(f"  JAMMED sigma    : uniform[{JAMMED_SIGMA_MIN:.2f}, {JAMMED_SIGMA_MAX:.2f}]")
print(f"  JAMMED types    : {', '.join(JAMMER_TYPES)}")

section("Step 4 - Streaming BUSY samples")
t_start = time.time()
busy_collected = 0

with h5py.File(FILE_PATH, "r") as f_in:
    with h5py.File(OUTPUT_PATH, "w") as f_out:
        x_out = f_out.create_dataset(
            "X",
            shape=(0, seq_len, iq_dim),
            maxshape=(None, seq_len, iq_dim),
            dtype="float32",
            chunks=(256, seq_len, iq_dim),
            compression="gzip",
            compression_opts=4,
        )
        y_out = f_out.create_dataset("y", shape=(0,), maxshape=(None,), dtype="int8")
        snr_out = f_out.create_dataset("snr", shape=(0,), maxshape=(None,), dtype="float32")

        f_out.attrs["class_names"] = ["BUSY", "FREE", "JAMMED"]
        f_out.attrs["busy_definition"] = "all_modulations_high_snr"
        f_out.attrs["busy_snr_min"] = BUSY_SNR_MIN
        f_out.attrs["free_sigma_min"] = FREE_SIGMA_MIN
        f_out.attrs["free_sigma_max"] = FREE_SIGMA_MAX
        f_out.attrs["jammed_sigma_min"] = JAMMED_SIGMA_MIN
        f_out.attrs["jammed_sigma_max"] = JAMMED_SIGMA_MAX
        f_out.attrs["jammer_types"] = JAMMER_TYPES
        f_out.attrs["source_modulations"] = CLASSES

        for batch_start in range(0, total_samples, READ_BATCH):
            if busy_collected >= balance_target:
                break

            batch_end = min(batch_start + READ_BATCH, total_samples)
            z_batch = np.asarray(f_in["Z"][batch_start:batch_end], dtype=np.float32)
            snr_batch = z_batch[:, 0]
            batch_mask = snr_batch >= BUSY_SNR_MIN

            if not batch_mask.any():
                del z_batch, snr_batch, batch_mask
                continue

            x_batch = np.asarray(f_in["X"][batch_start:batch_end], dtype=np.float32)
            rows = x_batch[batch_mask]
            snr_rows = snr_batch[batch_mask]
            slots = min(rows.shape[0], balance_target - busy_collected)

            write_block(x_out, y_out, snr_out, rows[:slots], 0, snr_rows[:slots])
            busy_collected += slots

            del z_batch, snr_batch, batch_mask, x_batch, rows, snr_rows
            gc.collect()

            batch_num = batch_start // READ_BATCH
            if batch_num % 20 == 0 or busy_collected == balance_target:
                elapsed = time.time() - t_start
                print(f"  Batch {batch_num:4d} | {progress_bar(busy_collected, balance_target)} | {elapsed:.0f}s")

print(f"\n  BUSY collected : {busy_collected:,} in {time.time() - t_start:.1f}s")

section("Step 5 - Generating FREE samples")
t_free = time.time()
free_written = 0
free_sigma_sum = 0.0
free_sigma_sq_sum = 0.0

with h5py.File(OUTPUT_PATH, "a") as f_out:
    x_out = f_out["X"]
    y_out = f_out["y"]
    snr_out = f_out["snr"]

    while free_written < balance_target:
        this_batch = min(SYNTH_BATCH, balance_target - free_written)
        x_batch, snr_batch, sigmas = generate_free_batch(this_batch, seq_len, iq_dim)
        write_block(x_out, y_out, snr_out, x_batch, 1, snr_batch)

        free_written += this_batch
        free_sigma_sum += float(sigmas.sum())
        free_sigma_sq_sum += float(np.square(sigmas).sum())

        del x_batch, sigmas
        gc.collect()

        if free_written % 100_000 < SYNTH_BATCH or free_written == balance_target:
            print(f"  FREE   | {progress_bar(free_written, balance_target)}")

    f_out.attrs["free_sigma_mean"] = free_sigma_sum / balance_target
    f_out.attrs["free_sigma_std"] = (
        free_sigma_sq_sum / balance_target - (free_sigma_sum / balance_target) ** 2
    ) ** 0.5

print(f"\n  FREE written : {free_written:,} in {time.time() - t_free:.1f}s")

section("Step 6 - Generating JAMMED samples")
t_jammed = time.time()
jammed_written = 0
jammed_sigma_sum = 0.0
jammed_sigma_sq_sum = 0.0
jammer_type_counts = {name: 0 for name in JAMMER_TYPES}

with h5py.File(OUTPUT_PATH, "a") as f_out:
    x_out = f_out["X"]
    y_out = f_out["y"]
    snr_out = f_out["snr"]

    while jammed_written < balance_target:
        this_batch = min(SYNTH_BATCH, balance_target - jammed_written)
        x_batch, snr_batch, sigmas, jammer_names = generate_jammed_batch(this_batch, seq_len, iq_dim)
        write_block(x_out, y_out, snr_out, x_batch, 2, snr_batch)

        jammed_written += this_batch
        jammed_sigma_sum += float(sigmas.sum())
        jammed_sigma_sq_sum += float(np.square(sigmas).sum())
        for jammer_name in jammer_names:
            jammer_type_counts[jammer_name] += 1

        del x_batch, sigmas, jammer_names
        gc.collect()

        if jammed_written % 100_000 < SYNTH_BATCH or jammed_written == balance_target:
            print(f"  JAMMED | {progress_bar(jammed_written, balance_target)}")

    f_out.attrs["jammed_sigma_mean"] = jammed_sigma_sum / balance_target
    f_out.attrs["jammed_sigma_std"] = (
        jammed_sigma_sq_sum / balance_target - (jammed_sigma_sum / balance_target) ** 2
    ) ** 0.5
    for jammer_name, count in jammer_type_counts.items():
        f_out.attrs[f"jammed_count_{jammer_name}"] = count

print(f"\n  JAMMED written : {jammed_written:,} in {time.time() - t_jammed:.1f}s")

section("Step 7 - Verification")
sample_probe = 500
banner("TASK 1.4 - VERIFICATION REPORT")

with h5py.File(OUTPUT_PATH, "r") as f:
    total = int(f["y"].shape[0])

    print(f"\n  Output file : {OUTPUT_PATH}")
    print(f"  X shape     : {f['X'].shape}")
    print(f"  Total rows  : {total:,}")
    print("  Verification pass is streamed to avoid high RAM usage.")

stats = stream_output_stats(OUTPUT_PATH, sample_probe)

print(f"\n  {'Class':<10} {'Count':>12} {'Percent':>10}")
print(f"  {'-' * 36}")
for label in [0, 1, 2]:
    count = stats["class_counts"][label]
    pct = count / stats["total"] * 100.0 if stats["total"] else 0.0
    print(f"  {LABEL_NAMES[label]:<10} {count:>12,} {pct:>9.2f}%")

print(f"\n  Mean signal energy probe ({sample_probe} samples per class):")
print(f"  {'-' * 48}")
for label in [0, 1, 2]:
    print(f"  {LABEL_NAMES[label]:<10} {stats['energies'][label]:>10.4f}")

print("\n  Energy ordering check:")
free_e = stats["energies"][1]
busy_e = stats["energies"][0]
jammed_e = stats["energies"][2]
print(f"    FREE ({free_e:.4f}) < BUSY ({busy_e:.4f})   -> {'OK' if free_e < busy_e else 'FAIL'}")
print(f"    BUSY ({busy_e:.4f}) < JAMMED ({jammed_e:.4f}) -> {'OK' if busy_e < jammed_e else 'FAIL'}")

print(f"\n  {'Class':<10} {'SNR min':>8} {'SNR mean':>10} {'SNR max':>8}")
print(f"  {'-' * 42}")
for label in [0, 1, 2]:
    print(
        f"  {LABEL_NAMES[label]:<10} "
        f"{stats['snr_min'][label]:>7.1f} "
        f"{stats['snr_mean'][label]:>10.1f} "
        f"{stats['snr_max'][label]:>8.1f}"
    )

with h5py.File(OUTPUT_PATH, "r") as f:
    print("\n  Stored generation metadata:")
    print(f"    busy_definition      : {f.attrs['busy_definition']}")
    print(f"    busy_snr_min         : {f.attrs['busy_snr_min']}")
    print(f"    free_sigma_range     : [{f.attrs['free_sigma_min']:.2f}, {f.attrs['free_sigma_max']:.2f}]")
    print(f"    jammed_sigma_range   : [{f.attrs['jammed_sigma_min']:.2f}, {f.attrs['jammed_sigma_max']:.2f}]")
    print(f"    jammer_types         : {list(f.attrs['jammer_types'])}")

total_time = time.time() - t_start
banner(f"Task 1.4 Complete - total time: {total_time / 60:.1f} min")
print()
