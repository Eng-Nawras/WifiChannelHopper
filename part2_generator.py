"""
ENCS5323 - Wireless and Mobile Networks
Project - Part 2: Dynamic Channel-Hopping Wi-Fi-like Interference Generator
Hardware: ADALM-PLUTO SDR (used for BOTH sensing and transmitting in standalone mode,
          or ONLY transmitting when run together with Part 1 on a second Pluto unit)

Author: <your name / student number>

HOW TO RUN:
    1. Connect the Pluto to your PC via USB.
    2. pip install pyadi-iio numpy
    3. python part2_generator.py
    (Run this on your own machine, NOT in a cloud sandbox - it needs the physical device.)
"""

import time
import numpy as np
import adi  # from pyadi-iio

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------

PLUTO_URI = "usb:1.4.5"          # default Pluto IP over USB; change if needed

BAND_START = 2400e6                    # Hz - start of 2.4 GHz ISM band
BAND_STOP = 2500e6                     # Hz - end of 2.4 GHz ISM band

# Standard Wi-Fi 2.4 GHz channel centre frequencies (channels 1-13, 20 MHz each)
WIFI_CHANNELS = {ch: 2412e6 + (ch - 1) * 5e6 for ch in range(1, 14)}
CHANNEL_BW = 20e6                      # Hz - Wi-Fi channel bandwidth

SENSE_SAMPLE_RATE = 61.44e6            # Hz - RX sample rate used while scanning
SENSE_NFFT = 4096                      # FFT size for the sensing spectrum
SENSE_DURATION = 0.05                  # seconds of samples per sensing snapshot

OCCUPANCY_THRESHOLD_DB = -55           # power (dBFS-ish) above which a channel is "occupied"
                                        # -> calibrate this against a known-idle channel first

TX_BANDWIDTH = 20e6                    # Hz - generated signal bandwidth (match a Wi-Fi channel)
TX_GAIN = -45                          # dB - keep LOW; see safety notice in the assignment
                                        # (-45 dB for safe initial bench testing; raise deliberately later)
RE_EVALUATE_PERIOD = 2.0               # seconds between re-sensing / possible hops

# ----------------------------------------------------------------------------
# 2. CONNECT TO PLUTO
# ----------------------------------------------------------------------------

def connect_pluto(uri=PLUTO_URI):
    sdr = adi.Pluto(uri)
    sdr.rx_lo = int(2450e6)             # will be re-tuned per scan step
    sdr.rx_rf_bandwidth = int(SENSE_SAMPLE_RATE)
    sdr.rx_sample_rate = int(SENSE_SAMPLE_RATE)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 30
    sdr.rx_buffer_size = int(SENSE_SAMPLE_RATE * SENSE_DURATION)

    sdr.tx_lo = int(2437e6)             # placeholder, set properly before each hop
    sdr.tx_rf_bandwidth = int(TX_BANDWIDTH)
    sdr.tx_sample_rate = int(TX_BANDWIDTH)
    sdr.tx_hardwaregain_chan0 = TX_GAIN
    return sdr


# ----------------------------------------------------------------------------
# 3. SENSING: figure out which Wi-Fi channels are occupied right now
# ----------------------------------------------------------------------------

def scan_band_power(sdr):
    """
    Sweeps the RX front-end across the 2.4 GHz band in steps equal to the
    Pluto's instantaneous sample rate, captures IQ samples, and returns
    a (freqs, power_db) array covering BAND_START..BAND_STOP.

    NOTE: Pluto's max RX bandwidth (~56-61 MHz) can't capture the whole
    100 MHz band in a single snapshot, so we step the LO across a couple
    of sub-bands and stitch the spectra together.
    """
    step = SENSE_SAMPLE_RATE * 0.9      # slight overlap between steps
    centers = np.arange(BAND_START + step / 2, BAND_STOP, step)

    all_freqs = []
    all_power = []

    for center in centers:
        sdr.rx_lo = int(center)
        time.sleep(0.01)                # let the LO settle
        samples = sdr.rx()

        # Welch/periodogram-style PSD
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
    """
    For each standard Wi-Fi channel, average the power inside its 20 MHz
    span and flag it as occupied if above OCCUPANCY_THRESHOLD_DB.
    Returns a list of occupied channel numbers, e.g. [1, 6, 11].
    """
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
# 4. CHANNEL SELECTION: pick the channel that maximizes distance from
#    every currently occupied channel
# ----------------------------------------------------------------------------

def choose_best_channel(occupied_channels, all_channels=None):
    """
    Chooses the channel (from all_channels) whose MINIMUM distance to any
    occupied channel is MAXIMIZED (a classic max-min / "largest gap" rule).
    If nothing is occupied, defaults to a central channel (e.g. 6).
    """
    if all_channels is None:
        all_channels = list(WIFI_CHANNELS.keys())

    if not occupied_channels:
        return 6  # reasonable default when the band is empty

    best_channel = None
    best_min_distance = -1

    for candidate in all_channels:
        # distance in MHz between channel centers
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

def generate_wifi_like_signal(sample_rate=TX_BANDWIDTH, duration=0.01):
    """
    Generates a simple wideband, noise-like complex baseband signal that
    occupies ~20 MHz, similar in spectral shape to an OFDM Wi-Fi signal.
    (A full 802.11 PHY isn't required - the assignment asks for a
    Wi-Fi-LIKE signal matching channel BW/frequency, not a compliant frame.)
    """
    n = int(sample_rate * duration)
    # Complex white noise gives a flat, wideband spectrum across TX_BANDWIDTH
    real = np.random.normal(0, 1, n)
    imag = np.random.normal(0, 1, n)
    iq = (real + 1j * imag).astype(np.complex64)

    # normalize to avoid clipping the DAC
    iq /= np.max(np.abs(iq))
    iq *= 2 ** 14  # scale for Pluto's expected sample range

    return iq


# ----------------------------------------------------------------------------
# 6. MAIN LOOP: sense -> choose -> hop -> transmit -> repeat
# ----------------------------------------------------------------------------

def main():
    sdr = connect_pluto()
    current_channel = None
    log = []  # keep (timestamp, occupied, chosen_channel) for your report

    try:
        while True:
            t0 = time.time()

            # --- 1. Sense ---
            freqs, power_db = scan_band_power(sdr)
            occupied = get_occupied_channels(freqs, power_db)

            # --- 2. Choose best channel ---
            # Exclude the channel we're currently transmitting on: our own TX
            # leaking into the RX front end during sensing can make it look
            # "occupied", which would otherwise cause the algorithm to hop
            # away from its own signal every cycle instead of settling.
            occupied_for_selection = [ch for ch in occupied if ch != current_channel]
            new_channel = choose_best_channel(occupied_for_selection)

            # --- 3. Hop if needed ---
            if new_channel != current_channel:
                sdr.tx_destroy_buffer()
                sdr.tx_lo = int(WIFI_CHANNELS[new_channel])
                iq_signal = generate_wifi_like_signal()
                sdr.tx_cyclic_buffer = True
                sdr.tx(iq_signal)
                current_channel = new_channel
                print(f"[HOP] Occupied={occupied} -> Selected channel {new_channel} "
                      f"({WIFI_CHANNELS[new_channel]/1e6:.1f} MHz)")

            # --- 4. Log for the report (reaction time, accuracy, etc.) ---
            log.append({
                "time": time.time(),
                "occupied_channels": occupied,
                "chosen_channel": current_channel,
                "reaction_time_s": time.time() - t0,
            })

            # --- 5. Wait before re-evaluating ---
            time.sleep(max(0, RE_EVALUATE_PERIOD - (time.time() - t0)))

    except KeyboardInterrupt:
        print("Stopping generator...")
    finally:
        sdr.tx_destroy_buffer()
        # Optional: dump `log` to a CSV/JSON here for your report plots
        # e.g. pandas.DataFrame(log).to_csv("part2_log.csv", index=False)


if __name__ == "__main__":
    main()
