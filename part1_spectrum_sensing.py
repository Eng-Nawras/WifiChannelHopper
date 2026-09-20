"""
ENCS5323 - Part 1: Live Wi-Fi Channel Occupancy Sensing (2.4 GHz) using ADALM-PLUTO
=====================================================================================

WHAT THIS SCRIPT DOES
----------------------
1. Connects to a PlutoSDR over IP (default ip:192.168.2.1).
2. Since the PlutoSDR cannot capture the full 100 MHz (2400-2500 MHz) in one
   shot, it RETUNES ("sweeps") across several center frequencies, capturing
   IQ samples at each one, computing a Welch PSD, and stitching the middle
   (clean) portion of each capture into one continuous spectrum covering the
   whole band.
3. Repeats this sweep every SWEEP_INTERVAL_S seconds, for N_SWEEPS times.
4. For each sweep, averages the power inside each of the 13 Wi-Fi channel
   bands (2412-2472 MHz, 5 MHz spacing, ~22 MHz wide) and decides whether
   that channel is "occupied" (power above noise floor + margin).
5. Produces:
     - full_spectrum_plot.png   -> power spectrum, full band, labeled axes
     - occupancy_heatmap.png    -> channel occupancy over time (heatmap)
     - sweep_data.npz           -> raw data saved for the report / reuse
     - occupancy_log.csv        -> per-sweep, per-channel power + occupied flag

HOW TO RUN
----------
    (venv active) > python part1_spectrum_sensing.py

Tune the CONFIG block below before running (sample rate, gain, sweep count,
threshold, interval). Everything else can be left alone.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import welch
import adi
import csv
from datetime import datetime

# ----------------------------- CONFIG ---------------------------------- #
PLUTO_URI          = "ip:192.168.3.1"   # this unit's custom IP (default would be 192.168.2.1)

BAND_START_HZ      = 2400e6             # bottom edge of 2.4 GHz ISM band
BAND_STOP_HZ       = 2500e6             # top edge

SAMPLE_RATE_HZ     = 30e6               # RX sample rate per capture (Hz)
RF_BANDWIDTH_HZ    = 30e6               # analog filter bandwidth, match sample rate
USABLE_FRACTION    = 0.8                # keep only the central 80% of each capture
                                         # (edges roll off due to the anti-alias filter)
N_SAMPLES          = 2**16              # IQ samples captured per center frequency
NPERSEG             = 1024              # Welch FFT segment length -> freq resolution

RX_GAIN_DB          = 40                # manual RX gain (0-70 dB typical for Pluto)
GAIN_MODE           = "manual"          # "manual" or "slow_attack" (AGC)

N_SWEEPS            = 40                # how many full-band sweeps to record
SWEEP_INTERVAL_S     = 3.0              # seconds between the START of each sweep

OCCUPANCY_MARGIN_DB  = 6.0              # dB above noise floor to call a channel "occupied"

# Standard 2.4 GHz Wi-Fi channel centers (MHz) and nominal channel width (MHz)
WIFI_CHANNELS_MHZ = {
    1: 2412, 2: 2417, 3: 2422, 4: 2427, 5: 2432, 6: 2437, 7: 2442,
    8: 2447, 9: 2452, 10: 2457, 11: 2462, 12: 2467, 13: 2472,
}
CHANNEL_WIDTH_MHZ = 22.0
# -------------------------------------------------------------------------- #


def build_center_frequencies():
    """Compute the list of LO center frequencies needed to cover the full band,
    using only the central USABLE_FRACTION of each capture."""
    usable_bw = SAMPLE_RATE_HZ * USABLE_FRACTION
    centers = []
    f = BAND_START_HZ + usable_bw / 2
    while f - usable_bw / 2 < BAND_STOP_HZ:
        centers.append(f)
        f += usable_bw
    return centers, usable_bw


def connect_pluto():
    sdr = adi.Pluto(PLUTO_URI)
    sdr.sample_rate = int(SAMPLE_RATE_HZ)
    sdr.rx_rf_bandwidth = int(RF_BANDWIDTH_HZ)
    sdr.rx_buffer_size = N_SAMPLES
    sdr.gain_control_mode_chan0 = GAIN_MODE
    if GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = RX_GAIN_DB
    return sdr


def capture_psd_at(sdr, center_hz):
    """Tune to center_hz, capture samples, return (freqs_hz, power_db) for that slice."""
    sdr.rx_lo = int(center_hz)
    time.sleep(0.05)          # let the synthesizer settle after retuning
    sdr.rx()                  # discard one buffer (flush stale samples in the pipeline)
    samples = sdr.rx()

    freqs, psd = welch(
        samples,
        fs=SAMPLE_RATE_HZ,
        nperseg=NPERSEG,
        return_onesided=False,
        scaling="density",
    )
    freqs = np.fft.fftshift(freqs) + center_hz
    psd = np.fft.fftshift(psd)
    power_db = 10 * np.log10(psd + 1e-20)   # relative power in dB (not calibrated dBm)

    # The AD9361 leaks a bit of the LO back into the received signal, which shows
    # up as an artificial spike exactly at the center (DC) frequency of this
    # capture. Notch it out and fill the gap by interpolating from its neighbors,
    # so it doesn't get mistaken for a real Wi-Fi signal.
    dc_bin_width = 3 * (SAMPLE_RATE_HZ / NPERSEG)
    notch_mask = np.abs(freqs - center_hz) < dc_bin_width
    if notch_mask.any() and not notch_mask.all():
        power_db[notch_mask] = np.interp(
            freqs[notch_mask], freqs[~notch_mask], power_db[~notch_mask]
        )

    return freqs, power_db


def do_one_sweep(sdr, centers, usable_bw):
    """Capture at every center frequency and stitch into one continuous spectrum."""
    all_freqs, all_power = [], []
    for center in centers:
        freqs, power_db = capture_psd_at(sdr, center)
        lo_edge = center - usable_bw / 2
        hi_edge = center + usable_bw / 2
        mask = (freqs >= lo_edge) & (freqs <= hi_edge)
        all_freqs.append(freqs[mask])
        all_power.append(power_db[mask])

    freqs_full = np.concatenate(all_freqs)
    power_full = np.concatenate(all_power)
    order = np.argsort(freqs_full)
    freqs_full, power_full = freqs_full[order], power_full[order]

    # The last hop can slightly overshoot BAND_STOP_HZ - trim back to the exact
    # requested band so the plot axis never extends past 2400-2500 MHz.
    band_mask = (freqs_full >= BAND_START_HZ) & (freqs_full <= BAND_STOP_HZ)
    return freqs_full[band_mask], power_full[band_mask]


def channel_powers(freqs_hz, power_db):
    """Average the stitched spectrum inside each Wi-Fi channel's bandwidth."""
    freqs_mhz = freqs_hz / 1e6
    result = {}
    for ch, center_mhz in WIFI_CHANNELS_MHZ.items():
        lo = center_mhz - CHANNEL_WIDTH_MHZ / 2
        hi = center_mhz + CHANNEL_WIDTH_MHZ / 2
        mask = (freqs_mhz >= lo) & (freqs_mhz <= hi)
        result[ch] = np.mean(power_db[mask]) if np.any(mask) else np.nan
    return result


def setup_live_plots():
    """Create the two figures once, before the sweep loop, and return handles
    to the pieces we'll update every sweep (so we don't redraw from scratch)."""
    plt.ion()  # interactive mode: figures update without blocking the script

    # --- Figure 1: power spectrum, redrawn every sweep ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    (line,) = ax1.plot([], [], linewidth=0.7)
    for ch, center in WIFI_CHANNELS_MHZ.items():
        ax1.axvline(center, color="gray", linestyle=":", linewidth=0.5)
    ax1.set_xlim(BAND_START_HZ / 1e6, BAND_STOP_HZ / 1e6)
    ax1.set_ylim(-70, 5)
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Power (dB, relative)")
    ax1.set_title("2.4 GHz Band Power Spectrum (live)")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    # --- Figure 2: occupancy heatmap, grows one row per sweep ---
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    blank = np.full((N_SWEEPS, len(WIFI_CHANNELS_MHZ)), np.nan)
    im = ax2.imshow(blank, aspect="auto", origin="lower",
                     extent=[0.5, len(WIFI_CHANNELS_MHZ) + 0.5, 0, N_SWEEPS],
                     cmap="viridis", vmin=-70, vmax=0)
    fig2.colorbar(im, ax=ax2, label="Power (dB, relative)")
    ax2.set_xticks(list(WIFI_CHANNELS_MHZ.keys()))
    ax2.set_xlabel("Wi-Fi Channel")
    ax2.set_ylabel("Sweep index (time -->)")
    ax2.set_title("Channel Occupancy Heatmap (live)")
    fig2.tight_layout()

    plt.show(block=False)
    return fig1, ax1, line, fig2, ax2, im


def main():
    centers, usable_bw = build_center_frequencies()
    print(f"Sweeping {len(centers)} center frequencies to cover "
          f"{BAND_START_HZ/1e6:.0f}-{BAND_STOP_HZ/1e6:.0f} MHz:")
    print([f"{c/1e6:.1f} MHz" for c in centers])

    sdr = connect_pluto()
    fig1, ax1, line, fig2, ax2, im = setup_live_plots()

    heatmap_power = np.full((N_SWEEPS, len(WIFI_CHANNELS_MHZ)), np.nan)
    heatmap_occupied = np.zeros((N_SWEEPS, len(WIFI_CHANNELS_MHZ)), dtype=bool)
    timestamps = []
    last_full_spectrum = None  # keep the final sweep for the saved PSD plot

    with open("occupancy_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "channel", "power_db", "occupied"])

        for i in range(N_SWEEPS):
            t0 = time.time()
            ts = datetime.now()
            freqs_hz, power_db = do_one_sweep(sdr, centers, usable_bw)
            last_full_spectrum = (freqs_hz, power_db)

            ch_powers = channel_powers(freqs_hz, power_db)
            noise_floor = np.nanpercentile(list(ch_powers.values()), 20)  # robust "quiet" estimate

            for j, ch in enumerate(WIFI_CHANNELS_MHZ.keys()):
                p = ch_powers[ch]
                occ = p > (noise_floor + OCCUPANCY_MARGIN_DB)
                heatmap_power[i, j] = p
                heatmap_occupied[i, j] = occ
                writer.writerow([ts.isoformat(), ch, f"{p:.2f}", occ])
            f.flush()  # so occupancy_log.csv is readable even if the run is interrupted

            timestamps.append(ts)
            occupied_list = [ch for ch, occ in zip(WIFI_CHANNELS_MHZ, heatmap_occupied[i]) if occ]
            print(f"[{ts.strftime('%H:%M:%S')}] sweep {i+1}/{N_SWEEPS} "
                  f"-> occupied channels: {occupied_list if occupied_list else 'none'}")

            # ---- update the live spectrum plot ----
            line.set_data(freqs_hz / 1e6, power_db)
            ax1.set_title(f"2.4 GHz Band Power Spectrum (live) \u2014 sweep {i+1}/{N_SWEEPS}")
            fig1.canvas.draw_idle()

            # ---- update the live heatmap (redraw only the filled rows) ----
            im.set_data(heatmap_power)
            fig2.canvas.draw_idle()

            elapsed = time.time() - t0
            sleep_left = SWEEP_INTERVAL_S - elapsed
            if sleep_left > 0:
                plt.pause(sleep_left)   # keeps both windows responsive while waiting
            else:
                plt.pause(0.001)        # still let the GUI event loop process the redraw

    # ---- Save raw data for the report ----
    np.savez("sweep_data.npz",
             heatmap_power=heatmap_power,
             heatmap_occupied=heatmap_occupied,
             channels=list(WIFI_CHANNELS_MHZ.keys()),
             last_spectrum_freqs=last_full_spectrum[0],
             last_spectrum_power=last_full_spectrum[1])

    # ---- Save final versions of both figures as PNGs ----
    fig1.savefig("full_spectrum_plot.png", dpi=150)
    print("Saved full_spectrum_plot.png")
    fig2.savefig("occupancy_heatmap.png", dpi=150)
    print("Saved occupancy_heatmap.png")

    print("Done. Close the plot windows to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()