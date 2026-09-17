"""
ENCS5323 - Wireless and Mobile Networks
Project - Part 2: Dynamic Channel-Hopping Wi-Fi-like Interference Generator
Hardware: ADALM-PLUTO SDR (used for BOTH sensing and transmitting in standalone mode,
          or ONLY transmitting when run together with Part 1 on a second Pluto unit)

Author: <your name / student number>

HOW TO RUN (basic):
    python part2_generator.py

HOW TO RUN (for the systematic experiments the assignment asks for):
    python part2_generator.py --bandwidth 10e6 --duration 40
    python part2_generator.py --bandwidth 20e6 --duration 40
    python part2_generator.py --center 2437e6 --duration 40   (force a fixed TX channel instead of auto-hop)

Every run appends one row per sensing cycle to logs/part2_log.csv, which is
exactly the data needed for the report: chosen channel, occupied channels,
reaction time, and the TX settings used for that run.
"""

import argparse
import csv
import os
import time

import numpy as np
import adi  # from pyadi-iio

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------

PLUTO_URI = "usb:1.4.5"                # direct USB connection (from `iio_info -s`)
                                        # NOTE: this address can change if you unplug/replug
                                        # the device or use a different port/PC. Re-run
                                        # `iio_info -s` and update this string if connection fails.

BAND_START = 2400e6                    # Hz - start of 2.4 GHz ISM band
BAND_STOP = 2500e6                     # Hz - end of 2.4 GHz ISM band

# Standard Wi-Fi 2.4 GHz channel centre frequencies (channels 1-13, 20 MHz each)
WIFI_CHANNELS = {ch: 2412e6 + (ch - 1) * 5e6 for ch in range(1, 14)}
CHANNEL_BW = 20e6                      # Hz - Wi-Fi channel bandwidth (for sensing bins)

SENSE_SAMPLE_RATE = 61.44e6            # Hz - RX sample rate used while scanning
SENSE_NFFT = 4096                      # FFT size for the sensing spectrum
SENSE_DURATION = 0.05                  # seconds of samples per sensing snapshot

OCCUPANCY_THRESHOLD_DB = -55           # calibrated against real measurements:
                                        # idle channels ~ -57 to -62 dB, Tareq network
                                        # (ch 8-11) ~ -49 to -52 dB on this bench setup.

DEFAULT_TX_BANDWIDTH = 20e6            # Hz - default generated signal bandwidth
TX_GAIN = -45                          # dB - safe initial bench testing level
RE_EVALUATE_PERIOD = 2.0               # seconds between re-sensing / possible hops

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "part2_log.csv")

# ----------------------------------------------------------------------------
# 2. CONNECT TO PLUTO
# ----------------------------------------------------------------------------

def connect_pluto(uri=PLUTO_URI, tx_bandwidth=DEFAULT_TX_BANDWIDTH):
    sdr = adi.Pluto(uri)
    sdr.rx_lo = int(2450e6)             # will be re-tuned per scan step
    sdr.rx_rf_bandwidth = int(SENSE_SAMPLE_RATE)
    sdr.rx_sample_rate = int(SENSE_SAMPLE_RATE)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 30
    sdr.rx_buffer_size = int(SENSE_SAMPLE_RATE * SENSE_DURATION)

    sdr.tx_lo = int(2437e6)             # placeholder, set properly before each hop
    sdr.tx_rf_bandwidth = int(tx_bandwidth)
    sdr.tx_sample_rate = int(tx_bandwidth)
    sdr.tx_hardwaregain_chan0 = TX_GAIN
    return sdr


# ----------------------------------------------------------------------------
# 3. SENSING: figure out which Wi-Fi channels are occupied right now
# ----------------------------------------------------------------------------

def scan_band_power(sdr):
    step = SENSE_SAMPLE_RATE * 0.9      # slight overlap between steps
    centers = np.arange(BAND_START + step / 2, BAND_STOP, step)

    all_freqs = []
    all_power = []

    for center in centers:
        sdr.rx_lo = int(center)
        time.sleep(0.01)                # let the LO settle
        samples = sdr.rx()

        window = np.hanning(len(samples))
        spectrum = np.fft.fftshift(np.fft.fft(samples * window, n=SENSE_NFFT))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
        freqs = center + np.fft.fftshift(
            np.fft.fftfreq(SENSE_NFFT, d=1 / SENSE_SAMPLE_RATE)
        )

        all_freqs.append(freqs)
        all_power.append(power_db)

    freqs = np.concatenate(all_freqs)
    power_db = np.concatenate(all_power)
    order = np.argsort(freqs)
    return freqs[order], power_db[order]


def get_occupied_channels(freqs, power_db):
    occupied = []
    for ch, f_center in WIFI_CHANNELS.items():
        mask = (freqs >= f_center - CHANNEL_BW / 2) & (freqs <= f_center + CHANNEL_BW / 2)
        if not np.any(mask):
            continue
        avg_power = np.mean(power_db[mask])
        if avg_power > OCCUPANCY_THRESHOLD_DB:
            occupied.append(ch)
    return occupied


# ----------------------------------------------------------------------------
# 4. CHANNEL SELECTION: max-min distance from occupied channels
# ----------------------------------------------------------------------------

def choose_best_channel(occupied_channels, all_channels=None):
    if all_channels is None:
        all_channels = list(WIFI_CHANNELS.keys())

    if not occupied_channels:
        return 6  # reasonable default when the band is empty

    best_channel = None
    best_min_distance = -1

    for candidate in all_channels:
        distances = [
            abs(WIFI_CHANNELS[candidate] - WIFI_CHANNELS[occ]) / 1e6
            for occ in occupied_channels
        ]
        min_distance = min(distances)

        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_channel = candidate

    return best_channel


# ----------------------------------------------------------------------------
# 5. SIGNAL GENERATION: build a Wi-Fi-like baseband waveform
# ----------------------------------------------------------------------------

def generate_wifi_like_signal(sample_rate, duration=0.01):
    n = int(sample_rate * duration)
    real = np.random.normal(0, 1, n)
    imag = np.random.normal(0, 1, n)
    iq = (real + 1j * imag).astype(np.complex64)
    iq /= np.max(np.abs(iq))
    iq *= 2 ** 14
    return iq


# ----------------------------------------------------------------------------
# 6. LOGGING
# ----------------------------------------------------------------------------

def init_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    is_new = not os.path.exists(LOG_FILE)
    f = open(LOG_FILE, "a", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow([
            "timestamp", "tx_bandwidth_hz", "forced_center_hz",
            "occupied_channels", "chosen_channel", "chosen_freq_hz",
            "hopped_this_cycle", "reaction_time_s",
        ])
    return f, writer


# ----------------------------------------------------------------------------
# 7. MAIN LOOP
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Part 2: Wi-Fi-like interference generator")
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_TX_BANDWIDTH,
                         help="TX signal bandwidth in Hz, e.g. 5e6, 10e6, 20e6 (default 20e6)")
    parser.add_argument("--center", type=float, default=None,
                         help="Force a fixed TX centre frequency in Hz instead of auto-hopping "
                              "(e.g. 2437e6). Sensing/logging still runs for comparison.")
    parser.add_argument("--duration", type=float, default=None,
                         help="Auto-stop after this many seconds (recommended: 30-60, "
                              "per the assignment's safety notice). Omit to run until Ctrl+C.")
    args = parser.parse_args()

    sdr = connect_pluto(tx_bandwidth=args.bandwidth)
    current_channel = None
    log_file, writer = init_log()
    start_time = time.time()

    print(f"Starting run: bandwidth={args.bandwidth/1e6:.1f} MHz, "
          f"forced_center={'auto-hop' if args.center is None else f'{args.center/1e6:.1f} MHz'}, "
          f"duration={'until Ctrl+C' if args.duration is None else f'{args.duration:.0f}s'}")

    try:
        while True:
            t0 = time.time()
            if args.duration is not None and (t0 - start_time) >= args.duration:
                print(f"Reached configured duration ({args.duration:.0f}s), stopping.")
                break

            # --- 1. Sense ---
            freqs, power_db = scan_band_power(sdr)
            occupied = get_occupied_channels(freqs, power_db)

            # --- 2. Choose channel (exclude the one we're currently transmitting on,
            #        to avoid self-interference biasing the decision) ---
            occupied_for_selection = [ch for ch in occupied if ch != current_channel]
            if args.center is not None:
                # fixed-frequency mode: pick the nearest standard channel just for logging
                new_channel = min(WIFI_CHANNELS, key=lambda c: abs(WIFI_CHANNELS[c] - args.center))
                tx_freq = args.center
            else:
                new_channel = choose_best_channel(occupied_for_selection)
                tx_freq = WIFI_CHANNELS[new_channel]

            # --- 3. Hop if needed ---
            hopped = new_channel != current_channel
            if hopped:
                sdr.tx_destroy_buffer()
                sdr.tx_lo = int(tx_freq)
                iq_signal = generate_wifi_like_signal(sample_rate=args.bandwidth)
                sdr.tx_cyclic_buffer = True
                sdr.tx(iq_signal)
                current_channel = new_channel
                print(f"[HOP] Occupied={occupied} -> Selected channel {new_channel} "
                      f"({tx_freq/1e6:.1f} MHz, bw={args.bandwidth/1e6:.1f} MHz)")

            # --- 4. Log every cycle (not just hops) for the report ---
            reaction_time = time.time() - t0
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), args.bandwidth,
                args.center if args.center is not None else "",
                ";".join(map(str, occupied)), current_channel, tx_freq,
                hopped, f"{reaction_time:.3f}",
            ])
            log_file.flush()

            # --- 5. Wait before re-evaluating ---
            time.sleep(max(0, RE_EVALUATE_PERIOD - reaction_time))

    except KeyboardInterrupt:
        print("Stopping generator (Ctrl+C)...")
    finally:
        sdr.tx_destroy_buffer()
        log_file.close()
        print(f"Log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
