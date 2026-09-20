"""
ENCS5323 - Wireless and Mobile Networks
Part 2: Dynamic Channel-Hopping Wi-Fi-like Signal Generator
ADALM-PLUTO SDR

Designed for the controlled indoor lab setup required by the assignment.

Main fixes in this version:
- Uses the Pluto device IP from the config file: 192.168.2.2
- Uses sdr.sample_rate (shared Pluto sample rate)
- Stops TX before sensing, so the Pluto does not classify its own TX as occupancy
- Uses the same sweep idea as Part 1 to cover 2400-2500 MHz
- Uses an adaptive noise-floor + margin occupancy decision
- Restarts TX after every sensing cycle, even if it stays on the same channel
- Generates a bandwidth-limited Wi-Fi-like wideband test waveform
- Logs occupied channels, selected channel, hopping, and reaction time
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
import adi


# ============================================================
# CONFIGURATION
# ============================================================

# From your Pluto config file:
#   ipaddr      = 192.168.2.2   -> Pluto device
#   ipaddr_host = 192.168.2.10  -> PC-side USB Ethernet address
PLUTO_URI = "ip:192.168.2.2"

BAND_START_HZ = 2400e6
BAND_STOP_HZ = 2500e6

# Practical receive settings, matching Part 1 closely
SAMPLE_RATE_HZ = 30e6
RF_BANDWIDTH_HZ = 30e6
USABLE_FRACTION = 0.8

N_SAMPLES = 2**16
NPERSEG = 1024

RX_GAIN_DB = 30
GAIN_MODE = "manual"

# Standard 2.4 GHz Wi-Fi channels 1-13
WIFI_CHANNELS_HZ = {
    1: 2412e6,
    2: 2417e6,
    3: 2422e6,
    4: 2427e6,
    5: 2432e6,
    6: 2437e6,
    7: 2442e6,
    8: 2447e6,
    9: 2452e6,
    10: 2457e6,
    11: 2462e6,
    12: 2467e6,
    13: 2472e6,
}

CHANNEL_WIDTH_HZ = 20e6

# Adaptive occupancy threshold:
# occupied if channel power > estimated noise floor + this margin
OCCUPANCY_MARGIN_DB = 6.0

DEFAULT_TX_BANDWIDTH_HZ = 20e6

# Keep TX weak for controlled bench/lab testing
TX_GAIN_DB = -45

# Re-sense / re-evaluate every 2 seconds
RE_EVALUATE_PERIOD_S = 2.0

# Give RX a moment after stopping TX
TX_TO_RX_SETTLE_S = 0.10

# Assignment recommends short test runs
DEFAULT_DURATION_S = 40.0
MAX_DURATION_S = 60.0

TX_BUFFER_SAMPLES = 65536

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "part2_log.csv"


# ============================================================
# SWEEP FREQUENCIES
# ============================================================

def build_center_frequencies():
    """
    Build the center frequencies needed to cover 2400-2500 MHz.
    Only the central part of each RX capture is kept.
    """
    usable_bw = SAMPLE_RATE_HZ * USABLE_FRACTION

    centers = []
    f = BAND_START_HZ + usable_bw / 2

    while f - usable_bw / 2 < BAND_STOP_HZ:
        centers.append(f)
        f += usable_bw

    return centers, usable_bw


# ============================================================
# CONNECT TO PLUTO
# ============================================================

def connect_pluto(tx_bandwidth_hz):
    print(f"Connecting to Part 2 Pluto at {PLUTO_URI} ...")

    sdr = adi.Pluto(PLUTO_URI)

    # Pluto uses one common sample-rate setting for RX/TX
    sdr.sample_rate = int(SAMPLE_RATE_HZ)

    # RX
    sdr.rx_rf_bandwidth = int(RF_BANDWIDTH_HZ)
    sdr.rx_buffer_size = int(N_SAMPLES)
    sdr.gain_control_mode_chan0 = GAIN_MODE

    if GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = RX_GAIN_DB

    # TX
    sdr.tx_rf_bandwidth = int(tx_bandwidth_hz)
    sdr.tx_hardwaregain_chan0 = TX_GAIN_DB
    sdr.tx_cyclic_buffer = True

    # Placeholder; changed before every TX start
    sdr.tx_lo = int(2437e6)

    print("Part 2 Pluto connected.")
    return sdr


# ============================================================
# TX CONTROL
# ============================================================

def stop_tx(sdr):
    """Stop any active cyclic TX buffer."""
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass


def generate_wifi_like_signal(
    sample_rate_hz,
    bandwidth_hz,
    n_samples=TX_BUFFER_SAMPLES
):
    """
    Generate a controlled wideband pseudo-OFDM / Wi-Fi-like waveform.

    It is NOT a decodable 802.11 frame. It is a wideband test signal
    with energy limited to approximately the requested bandwidth.
    """

    if bandwidth_hz <= 0:
        raise ValueError("TX bandwidth must be greater than 0.")

    if bandwidth_hz >= sample_rate_hz:
        raise ValueError(
            f"TX bandwidth must be below sample rate "
            f"({sample_rate_hz / 1e6:.1f} MHz)."
        )

    freqs = np.fft.fftfreq(
        n_samples,
        d=1.0 / sample_rate_hz
    )

    # Use 90% of the requested bandwidth to leave guard space
    active = np.abs(freqs) <= (0.45 * bandwidth_hz)

    spectrum = np.zeros(
        n_samples,
        dtype=np.complex64
    )

    rng = np.random.default_rng()

    count = np.count_nonzero(active)

    # QPSK-like random frequency-domain values
    re = rng.choice([-1.0, 1.0], size=count)
    im = rng.choice([-1.0, 1.0], size=count)

    spectrum[active] = (
        re + 1j * im
    ).astype(np.complex64)

    # Remove DC carrier component
    dc_index = np.argmin(np.abs(freqs))
    spectrum[dc_index] = 0

    iq = np.fft.ifft(spectrum).astype(np.complex64)

    peak = np.max(np.abs(iq))

    if peak > 0:
        iq /= peak

    # Conservative digital amplitude
    iq *= (0.5 * (2**14))

    return iq.astype(np.complex64)


def start_tx(sdr, tx_freq_hz, tx_bandwidth_hz):
    """Start cyclic transmission on the selected center frequency."""

    sdr.tx_lo = int(tx_freq_hz)
    sdr.tx_rf_bandwidth = int(tx_bandwidth_hz)
    sdr.tx_cyclic_buffer = True

    iq_signal = generate_wifi_like_signal(
        sample_rate_hz=SAMPLE_RATE_HZ,
        bandwidth_hz=tx_bandwidth_hz
    )

    sdr.tx(iq_signal)


# ============================================================
# SENSING
# ============================================================

def capture_psd_at(sdr, center_hz):
    """
    Tune RX to one center frequency and return
    frequency + relative PSD in dB.
    """

    sdr.rx_lo = int(center_hz)

    # Allow LO to settle
    time.sleep(0.05)

    # Flush one stale buffer
    sdr.rx()

    samples = sdr.rx()

    freqs, psd = welch(
        samples,
        fs=SAMPLE_RATE_HZ,
        nperseg=NPERSEG,
        return_onesided=False,
        scaling="density"
    )

    freqs = np.fft.fftshift(freqs) + center_hz
    psd = np.fft.fftshift(psd)

    power_db = 10 * np.log10(
        psd + 1e-20
    )

    # Remove center-frequency / DC artifact
    dc_bin_width = 3 * (
        SAMPLE_RATE_HZ / NPERSEG
    )

    notch_mask = (
        np.abs(freqs - center_hz) <
        dc_bin_width
    )

    if notch_mask.any() and not notch_mask.all():
        power_db[notch_mask] = np.interp(
            freqs[notch_mask],
            freqs[~notch_mask],
            power_db[~notch_mask]
        )

    return freqs, power_db


def scan_band_power(sdr, centers, usable_bw):
    """
    Sweep across 2400-2500 MHz and stitch the useful
    center parts of the captures together.
    """

    all_freqs = []
    all_power = []

    for center in centers:
        freqs, power_db = capture_psd_at(
            sdr,
            center
        )

        lo_edge = center - usable_bw / 2
        hi_edge = center + usable_bw / 2

        mask = (
            (freqs >= lo_edge) &
            (freqs <= hi_edge)
        )

        all_freqs.append(freqs[mask])
        all_power.append(power_db[mask])

    freqs_full = np.concatenate(all_freqs)
    power_full = np.concatenate(all_power)

    order = np.argsort(freqs_full)

    freqs_full = freqs_full[order]
    power_full = power_full[order]

    band_mask = (
        (freqs_full >= BAND_START_HZ) &
        (freqs_full <= BAND_STOP_HZ)
    )

    return (
        freqs_full[band_mask],
        power_full[band_mask]
    )


def calculate_channel_powers(freqs_hz, power_db):
    """
    Average measured power inside each 20 MHz
    Wi-Fi channel region.
    """

    channel_power = {}

    for ch, center_hz in WIFI_CHANNELS_HZ.items():

        lo = center_hz - CHANNEL_WIDTH_HZ / 2
        hi = center_hz + CHANNEL_WIDTH_HZ / 2

        mask = (
            (freqs_hz >= lo) &
            (freqs_hz <= hi)
        )

        if np.any(mask):
            channel_power[ch] = float(
                np.mean(power_db[mask])
            )
        else:
            channel_power[ch] = np.nan

    return channel_power


def get_occupied_channels(channel_power):
    """
    Estimate the noise floor from the current channel powers.
    A channel is occupied when it is sufficiently above that floor.
    """

    values = np.array(
        list(channel_power.values()),
        dtype=float
    )

    noise_floor = float(
        np.nanpercentile(values, 20)
    )

    threshold = (
        noise_floor + OCCUPANCY_MARGIN_DB
    )

    occupied = [
        ch
        for ch, p in channel_power.items()
        if np.isfinite(p) and p > threshold
    ]

    return occupied, noise_floor, threshold


# ============================================================
# CHANNEL SELECTION
# ============================================================

def choose_best_channel(occupied_channels):
    """
    Choose the standard Wi-Fi channel whose center frequency
    has the greatest minimum distance from all occupied channels.
    """

    all_channels = list(
        WIFI_CHANNELS_HZ.keys()
    )

    if not occupied_channels:
        return 6

    best_channel = None
    best_min_distance = -1.0

    for candidate in all_channels:

        distances_mhz = [
            abs(
                WIFI_CHANNELS_HZ[candidate] -
                WIFI_CHANNELS_HZ[occ]
            ) / 1e6
            for occ in occupied_channels
        ]

        min_distance = min(
            distances_mhz
        )

        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_channel = candidate

    return best_channel


# ============================================================
# LOGGING
# ============================================================

def init_log():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    is_new = not LOG_FILE.exists()

    f = open(
        LOG_FILE,
        "a",
        newline=""
    )

    writer = csv.writer(f)

    if is_new:
        writer.writerow([
            "timestamp",
            "tx_bandwidth_hz",
            "forced_center_hz",
            "noise_floor_db",
            "occupancy_threshold_db",
            "occupied_channels",
            "chosen_channel",
            "chosen_freq_hz",
            "hopped_this_cycle",
            "reaction_time_s"
        ])

    return f, writer


# ============================================================
# ARGUMENT VALIDATION
# ============================================================

def validate_args(args):

    if args.bandwidth <= 0:
        raise ValueError(
            "Bandwidth must be greater than 0."
        )

    if args.bandwidth > 20e6:
        raise ValueError(
            "For this project, keep TX bandwidth at or below 20 MHz."
        )

    if args.bandwidth >= SAMPLE_RATE_HZ:
        raise ValueError(
            "Bandwidth must be below the 30 MHz sample rate."
        )

    if args.duration <= 0:
        raise ValueError(
            "Duration must be greater than 0."
        )

    if args.duration > MAX_DURATION_S:
        raise ValueError(
            f"Keep the controlled test at or below "
            f"{MAX_DURATION_S:.0f} seconds."
        )

    if args.center is not None:
        if not (
            WIFI_CHANNELS_HZ[1] <= args.center <= WIFI_CHANNELS_HZ[13]
        ):
            raise ValueError(
                "Forced center must stay between "
                "2412 MHz and 2472 MHz."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Part 2: Dynamic channel-hopping "
            "Wi-Fi-like signal generator"
        )
    )

    parser.add_argument(
        "--bandwidth",
        type=float,
        default=DEFAULT_TX_BANDWIDTH_HZ,
        help=(
            "TX bandwidth in Hz. "
            "Examples: 5e6, 10e6, 20e6."
        )
    )

    parser.add_argument(
        "--center",
        type=float,
        default=None,
        help=(
            "Force a fixed TX center frequency in Hz "
            "instead of auto-hop, e.g. 2437e6."
        )
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help=(
            "Controlled run duration in seconds. "
            "Default 40, maximum 60."
        )
    )

    args = parser.parse_args()
    validate_args(args)

    centers, usable_bw = (
        build_center_frequencies()
    )

    print()
    print("======================================")
    print("Part 2 - Dynamic Channel Hopping")
    print("======================================")
    print(f"Pluto URI: {PLUTO_URI}")
    print(
        f"TX bandwidth: "
        f"{args.bandwidth / 1e6:.1f} MHz"
    )
    print(
        f"Mode: "
        f"{'AUTO-HOP' if args.center is None else 'FIXED'}"
    )
    print(
        f"Duration: {args.duration:.0f} seconds"
    )
    print(
        "Sensing centers:",
        [
            f"{c / 1e6:.1f} MHz"
            for c in centers
        ]
    )
    print()

    sdr = connect_pluto(
        tx_bandwidth_hz=args.bandwidth
    )

    log_file, writer = init_log()

    current_channel = None
    start_time = time.time()

    try:
        while True:

            cycle_start = time.time()

            if (
                cycle_start - start_time
                >= args.duration
            ):
                print(
                    f"Reached configured duration "
                    f"({args.duration:.0f}s)."
                )
                break

            # ----------------------------------------------
            # 1. STOP TX BEFORE SENSING
            # ----------------------------------------------

            stop_tx(sdr)
            time.sleep(TX_TO_RX_SETTLE_S)

            # ----------------------------------------------
            # 2. SENSE FULL BAND
            # ----------------------------------------------

            freqs, power_db = scan_band_power(
                sdr,
                centers,
                usable_bw
            )

            channel_power = (
                calculate_channel_powers(
                    freqs,
                    power_db
                )
            )

            (
                occupied,
                noise_floor,
                threshold
            ) = get_occupied_channels(
                channel_power
            )

            # ----------------------------------------------
            # 3. SELECT CHANNEL
            # ----------------------------------------------

            if args.center is not None:

                tx_freq = float(
                    args.center
                )

                new_channel = min(
                    WIFI_CHANNELS_HZ,
                    key=lambda ch: abs(
                        WIFI_CHANNELS_HZ[ch] -
                        tx_freq
                    )
                )

            else:

                new_channel = (
                    choose_best_channel(
                        occupied
                    )
                )

                tx_freq = (
                    WIFI_CHANNELS_HZ[
                        new_channel
                    ]
                )

            hopped = (
                new_channel != current_channel
            )

            # ----------------------------------------------
            # 4. START TX AGAIN
            # ----------------------------------------------

            start_tx(
                sdr,
                tx_freq_hz=tx_freq,
                tx_bandwidth_hz=args.bandwidth
            )

            # ----------------------------------------------
            # 5. REPORT
            # ----------------------------------------------

            if hopped:
                status = "HOP"
            else:
                status = "STAY"

            print(
                f"[{status}] "
                f"Occupied={occupied} | "
                f"Noise={noise_floor:.1f} dB | "
                f"Threshold={threshold:.1f} dB | "
                f"Selected ch {new_channel} "
                f"({tx_freq / 1e6:.1f} MHz)"
            )

            current_channel = new_channel

            reaction_time = (
                time.time() -
                cycle_start
            )

            # ----------------------------------------------
            # 6. LOG
            # ----------------------------------------------

            writer.writerow([
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                args.bandwidth,
                (
                    args.center
                    if args.center is not None
                    else ""
                ),
                f"{noise_floor:.2f}",
                f"{threshold:.2f}",
                ";".join(
                    map(str, occupied)
                ),
                current_channel,
                tx_freq,
                hopped,
                f"{reaction_time:.3f}"
            ])

            log_file.flush()

            # ----------------------------------------------
            # 7. WAIT BEFORE NEXT RE-EVALUATION
            # ----------------------------------------------

            sleep_left = (
                RE_EVALUATE_PERIOD_S -
                reaction_time
            )

            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        print("Stopping Part 2...")

    finally:
        stop_tx(sdr)
        log_file.close()

        print()
        print("TX stopped.")
        print(
            f"Log saved to: {LOG_FILE}"
        )


if __name__ == "__main__":
    main()
