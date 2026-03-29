from utils.loaders.gas_loader import load_gas_batches
from utils.preprocessing import clean_batches, normalize_batches
from utils.stream import create_data_stream, get_batch_boundaries
from utils.windowing import create_sliding_windows

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


if __name__ == "__main__":
    main()