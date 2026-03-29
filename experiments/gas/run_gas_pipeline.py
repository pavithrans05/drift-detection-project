from utils.loaders.gas_loader import load_gas_batches
from utils.preprocessing import clean_batches, normalize_batches
from utils.stream import create_data_stream, get_batch_boundaries
from utils.windowing import create_sliding_windows

from drift_detection.baselines.ks_test import compute_ks_scores
from drift_detection.baselines.page_hinkley import compute_ph_scores
from drift_detection.baselines.adwin import compute_adwin_drift

from utils.visualization import plot_drift

DATA_PATH = "data/gas"


def main():
    # 1. Load
    batches = load_gas_batches(DATA_PATH)
    print(f"Loaded {len(batches)} batches")

    # 2. Clean
    batches = clean_batches(batches)

    # 3. Normalize
    batches, scaler = normalize_batches(batches)

    # 4. Create stream
    data_stream, batch_sizes = create_data_stream(batches)

    # 5. Boundaries
    boundaries = get_batch_boundaries(batch_sizes)

    print("Data Stream Shape:", data_stream.shape)
    print("Batch Sizes:", batch_sizes)
    print("Batch Boundaries (first 3):", boundaries[:3])

    # Sanity check
    print("Min:", data_stream.min())
    print("Max:", data_stream.max())
    print("Mean:", data_stream.mean())
    print("Std:", data_stream.std())

    # 6. Sliding window
    window_size = 50
    windows = create_sliding_windows(data_stream, window_size)

    print("Windows Shape:", windows.shape)

    # 7. KS Drift (reference-based)
    ref_size = 50
    ks_scores = compute_ks_scores(windows, ref_size=ref_size)
    print("KS Scores computed")

    # 8. Page-Hinkley
    ph_scores = compute_ph_scores(windows)
    print("PH Scores computed")

    # 9. ADWIN
    adwin_scores, adwin_flags = compute_adwin_drift(windows)
    print("ADWIN Drift points:", sum(adwin_flags))

    # 🔥 Align boundaries with window indices
    window_boundaries = [
        (
            max(0, start - window_size - ref_size),
            max(0, end - window_size - ref_size)
        )
        for start, end in boundaries
    ]

    # 10. Plot KS
    plot_drift(
        ks_scores,
        window_boundaries,
        title="KS Drift Detection (Reference Window)",
        filename="ks_drift.png"
    )

    # 11. Plot PH
    plot_drift(
        ph_scores,
        window_boundaries,
        title="Page-Hinkley Drift Signal",
        filename="ph_drift.png"
    )

    # 12. Plot ADWIN
    plot_drift(
        adwin_scores,
        window_boundaries,
        title="ADWIN Drift Signal",
        filename="adwin_drift.png"
    )


if __name__ == "__main__":
    main()