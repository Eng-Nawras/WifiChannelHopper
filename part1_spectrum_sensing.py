"""
ENCS5323 - Part 1
Live Wi-Fi Channel Occupancy Sensing (2.4 GHz)
using ADALM-PLUTO
"""

import time
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import adi


# ============================================================
# CONFIGURATION
# ============================================================

PLUTO_URI = "ip:192.168.3.1"

BAND_START_HZ = 2400e6
BAND_STOP_HZ = 2500e6

SAMPLE_RATE_HZ = 30e6
RF_BANDWIDTH_HZ = 30e6

# Keep only the middle 80% of every capture
USABLE_FRACTION = 0.8

N_SAMPLES = 2**16
NPERSEG = 1024

RX_GAIN_DB = 40
GAIN_MODE = "manual"

# Number of complete scans
N_SWEEPS = 20

# Time between sweeps
SWEEP_INTERVAL_S = 3.0

# Channel is occupied if power is this much above noise floor
OCCUPANCY_MARGIN_DB = 6.0


# Wi-Fi channels 1-13
WIFI_CHANNELS_MHZ = {
    1: 2412,
    2: 2417,
    3: 2422,
    4: 2427,
    5: 2432,
    6: 2437,
    7: 2442,
    8: 2447,
    9: 2452,
    10: 2457,
    11: 2462,
    12: 2467,
    13: 2472,
}

CHANNEL_WIDTH_MHZ = 22.0


# ============================================================
# OUTPUT FILES
# ============================================================

# Save everything in the same folder as this Python file
BASE_DIR = Path(__file__).resolve().parent

SPECTRUM_FILE = BASE_DIR / "full_spectrum_plot.png"
HEATMAP_FILE = BASE_DIR / "occupancy_heatmap.png"
LOG_FILE = BASE_DIR / "occupancy_log.csv"
DATA_FILE = BASE_DIR / "sweep_data.npz"


# ============================================================
# BUILD SWEEP FREQUENCIES
# ============================================================

def build_center_frequencies():
    """
    Calculate the center frequencies needed to scan
    the complete 2400-2500 MHz range.
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

def connect_pluto():
    print(f"Connecting to PlutoSDR at {PLUTO_URI}...")

    sdr = adi.Pluto(PLUTO_URI)

    sdr.sample_rate = int(SAMPLE_RATE_HZ)
    sdr.rx_rf_bandwidth = int(RF_BANDWIDTH_HZ)
    sdr.rx_buffer_size = N_SAMPLES

    sdr.gain_control_mode_chan0 = GAIN_MODE

    if GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = RX_GAIN_DB

    print("PlutoSDR connected.")

    return sdr


# ============================================================
# CAPTURE ONE FREQUENCY SECTION
# ============================================================

def capture_psd_at(sdr, center_hz):
    """
    Tune Pluto to one center frequency,
    capture IQ samples,
    calculate the PSD,
    and return frequency + power.
    """

    sdr.rx_lo = int(center_hz)

    # Give the SDR time to settle after changing frequency
    time.sleep(0.05)

    # Discard old samples
    sdr.rx()

    # Capture new samples
    samples = sdr.rx()

    # Calculate Power Spectral Density using Welch method
    freqs, psd = welch(
        samples,
        fs=SAMPLE_RATE_HZ,
        nperseg=NPERSEG,
        return_onesided=False,
        scaling="density"
    )

    # Shift spectrum so frequency 0 is in the middle
    freqs = np.fft.fftshift(freqs) + center_hz
    psd = np.fft.fftshift(psd)

    # Convert to relative dB
    power_db = 10 * np.log10(psd + 1e-20)

    # --------------------------------------------------------
    # Remove PlutoSDR center-frequency/DC spike
    # --------------------------------------------------------

    dc_bin_width = 3 * (SAMPLE_RATE_HZ / NPERSEG)

    notch_mask = np.abs(freqs - center_hz) < dc_bin_width

    if notch_mask.any() and not notch_mask.all():

        power_db[notch_mask] = np.interp(
            freqs[notch_mask],
            freqs[~notch_mask],
            power_db[~notch_mask]
        )

    return freqs, power_db


# ============================================================
# COMPLETE 2400-2500 MHz SWEEP
# ============================================================

def do_one_sweep(sdr, centers, usable_bw):
    """
    Scan all required center frequencies and
    combine them into one full spectrum.
    """

    all_freqs = []
    all_power = []

    for center in centers:

        freqs, power_db = capture_psd_at(sdr, center)

        # Keep only middle part of capture
        lo_edge = center - usable_bw / 2
        hi_edge = center + usable_bw / 2

        mask = (
            (freqs >= lo_edge) &
            (freqs <= hi_edge)
        )

        all_freqs.append(freqs[mask])
        all_power.append(power_db[mask])

    # Combine all frequency sections
    freqs_full = np.concatenate(all_freqs)
    power_full = np.concatenate(all_power)

    # Sort frequencies
    order = np.argsort(freqs_full)

    freqs_full = freqs_full[order]
    power_full = power_full[order]

    # Keep exactly 2400-2500 MHz
    band_mask = (
        (freqs_full >= BAND_START_HZ) &
        (freqs_full <= BAND_STOP_HZ)
    )

    return (
        freqs_full[band_mask],
        power_full[band_mask]
    )


# ============================================================
# CALCULATE POWER FOR EACH WIFI CHANNEL
# ============================================================

def channel_powers(freqs_hz, power_db):
    """
    Calculate average measured power
    inside each Wi-Fi channel.
    """

    freqs_mhz = freqs_hz / 1e6

    result = {}

    for ch, center_mhz in WIFI_CHANNELS_MHZ.items():

        lo = center_mhz - CHANNEL_WIDTH_MHZ / 2
        hi = center_mhz + CHANNEL_WIDTH_MHZ / 2

        mask = (
            (freqs_mhz >= lo) &
            (freqs_mhz <= hi)
        )

        if np.any(mask):
            result[ch] = np.mean(power_db[mask])
        else:
            result[ch] = np.nan

    return result


# ============================================================
# CREATE LIVE FIGURES
# ============================================================

def setup_live_plots():

    plt.ion()

    # --------------------------------------------------------
    # Figure 1 - Full Spectrum
    # --------------------------------------------------------

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    line, = ax1.plot(
        [],
        [],
        linewidth=0.7
    )

    # Draw Wi-Fi channel center frequencies
    for ch, center in WIFI_CHANNELS_MHZ.items():

        ax1.axvline(
            center,
            color="gray",
            linestyle=":",
            linewidth=0.5
        )

    ax1.set_xlim(
        BAND_START_HZ / 1e6,
        BAND_STOP_HZ / 1e6
    )

    ax1.set_ylim(-70, 5)

    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Power (dB, relative)")

    ax1.set_title(
        "2.4 GHz Band Power Spectrum (live)"
    )

    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()


    # --------------------------------------------------------
    # Figure 2 - Channel Heatmap
    # --------------------------------------------------------

    fig2, ax2 = plt.subplots(figsize=(9, 6))

    blank = np.full(
        (N_SWEEPS, len(WIFI_CHANNELS_MHZ)),
        np.nan
    )

    im = ax2.imshow(
        blank,
        aspect="auto",
        origin="lower",
        extent=[
            0.5,
            len(WIFI_CHANNELS_MHZ) + 0.5,
            0,
            N_SWEEPS
        ],
        cmap="viridis",
        vmin=-70,
        vmax=0
    )

    fig2.colorbar(
        im,
        ax=ax2,
        label="Power (dB, relative)"
    )

    ax2.set_xticks(
        list(WIFI_CHANNELS_MHZ.keys())
    )

    ax2.set_xlabel("Wi-Fi Channel")
    ax2.set_ylabel("Sweep index (time -->)")

    ax2.set_title(
        "Channel Occupancy Heatmap (live)"
    )

    fig2.tight_layout()

    plt.show(block=False)

    return (
        fig1,
        ax1,
        line,
        fig2,
        ax2,
        im
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # Calculate scanning frequencies
    # --------------------------------------------------------

    centers, usable_bw = build_center_frequencies()

    print()
    print("======================================")
    print("Part 1 - Wi-Fi Spectrum Sensing")
    print("======================================")

    print(
        f"Sweeping {len(centers)} center frequencies "
        f"to cover "
        f"{BAND_START_HZ / 1e6:.0f}-"
        f"{BAND_STOP_HZ / 1e6:.0f} MHz"
    )

    print(
        "Center frequencies:",
        [f"{c / 1e6:.1f} MHz" for c in centers]
    )

    print()
    print(f"Output folder: {BASE_DIR}")
    print()


    # --------------------------------------------------------
    # Connect SDR
    # --------------------------------------------------------

    sdr = connect_pluto()


    # --------------------------------------------------------
    # Create figures
    # --------------------------------------------------------

    (
        fig1,
        ax1,
        line,
        fig2,
        ax2,
        im
    ) = setup_live_plots()


    # --------------------------------------------------------
    # Arrays for measurements
    # --------------------------------------------------------

    heatmap_power = np.full(
        (
            N_SWEEPS,
            len(WIFI_CHANNELS_MHZ)
        ),
        np.nan
    )

    heatmap_occupied = np.zeros(
        (
            N_SWEEPS,
            len(WIFI_CHANNELS_MHZ)
        ),
        dtype=bool
    )

    timestamps = []

    last_full_spectrum = None


    # --------------------------------------------------------
    # Create CSV file
    # --------------------------------------------------------

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "channel",
            "power_db",
            "occupied"
        ])


        # ====================================================
        # MAIN SWEEP LOOP
        # ====================================================

        for i in range(N_SWEEPS):

            t0 = time.time()

            ts = datetime.now()


            # -----------------------------------------------
            # Scan full band
            # -----------------------------------------------

            freqs_hz, power_db = do_one_sweep(
                sdr,
                centers,
                usable_bw
            )

            last_full_spectrum = (
                freqs_hz,
                power_db
            )


            # -----------------------------------------------
            # Calculate channel powers
            # -----------------------------------------------

            ch_powers = channel_powers(
                freqs_hz,
                power_db
            )


            # -----------------------------------------------
            # Estimate noise floor
            # -----------------------------------------------

            noise_floor = np.nanpercentile(
                list(ch_powers.values()),
                20
            )


            # -----------------------------------------------
            # Check occupancy
            # -----------------------------------------------

            for j, ch in enumerate(
                WIFI_CHANNELS_MHZ.keys()
            ):

                p = ch_powers[ch]

                occ = (
                    p >
                    noise_floor +
                    OCCUPANCY_MARGIN_DB
                )

                heatmap_power[i, j] = p

                heatmap_occupied[i, j] = occ

                writer.writerow([
                    ts.isoformat(),
                    ch,
                    f"{p:.2f}",
                    occ
                ])

            # Write CSV immediately
            f.flush()


            # -----------------------------------------------
            # Print occupied channels
            # -----------------------------------------------

            timestamps.append(ts)

            occupied_list = [
                ch
                for ch, occ in zip(
                    WIFI_CHANNELS_MHZ.keys(),
                    heatmap_occupied[i]
                )
                if occ
            ]

            print(
                f"[{ts.strftime('%H:%M:%S')}] "
                f"sweep {i + 1}/{N_SWEEPS} "
                f"-> occupied channels: "
                f"{occupied_list if occupied_list else 'none'}"
            )


            # ===============================================
            # UPDATE LIVE SPECTRUM
            # ===============================================

            line.set_data(
                freqs_hz / 1e6,
                power_db
            )

            ax1.set_title(
                "2.4 GHz Band Power Spectrum (live) "
                f"— sweep {i + 1}/{N_SWEEPS}"
            )

            fig1.canvas.draw_idle()


            # ===============================================
            # UPDATE LIVE HEATMAP
            # ===============================================

            im.set_data(heatmap_power)

            fig2.canvas.draw_idle()


            # ===============================================
            # SAVE FIGURES AFTER EVERY SWEEP
            # ===============================================

            fig1.savefig(
                SPECTRUM_FILE,
                dpi=150
            )

            fig2.savefig(
                HEATMAP_FILE,
                dpi=150
            )

            print(
                "   Updated figures saved."
            )


            # ===============================================
            # WAIT UNTIL NEXT SWEEP
            # ===============================================

            elapsed = time.time() - t0

            sleep_left = (
                SWEEP_INTERVAL_S - elapsed
            )

            if sleep_left > 0:

                plt.pause(sleep_left)

            else:

                plt.pause(0.001)


    # ========================================================
    # SAVE RAW DATA
    # ========================================================

    np.savez(
        DATA_FILE,
        heatmap_power=heatmap_power,
        heatmap_occupied=heatmap_occupied,
        channels=list(
            WIFI_CHANNELS_MHZ.keys()
        ),
        last_spectrum_freqs=
            last_full_spectrum[0],
        last_spectrum_power=
            last_full_spectrum[1]
    )


    # ========================================================
    # FINAL SAVE
    # ========================================================

    fig1.savefig(
        SPECTRUM_FILE,
        dpi=150
    )

    fig2.savefig(
        HEATMAP_FILE,
        dpi=150
    )


    print()
    print("======================================")
    print("Experiment finished")
    print("======================================")

    print(
        f"Saved: {SPECTRUM_FILE.name}"
    )

    print(
        f"Saved: {HEATMAP_FILE.name}"
    )

    print(
        f"Saved: {DATA_FILE.name}"
    )

    print(
        f"Saved: {LOG_FILE.name}"
    )

    print()
    print(
        f"Files are located in:\n{BASE_DIR}"
    )

    print()
    print(
        "Done. Close the plot windows to exit."
    )


    plt.ioff()
    plt.show()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
