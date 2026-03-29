import numpy as np

def load_gas_batch(file_path):
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # ignore label
            features = [float(x.split(':')[1]) for x in parts[1:]]
            data.append(features)
    
    return np.array(data)


def load_gas_dataset(folder_path):
    batches = []
    
    for i in range(1, 11):
        path = f"{folder_path}/batch{i}.dat"
        batch = load_gas_batch(path)
        batches.append(batch)
    
    X = np.vstack(batches)
    
    # meta: drift points
    batch_sizes = [len(b) for b in batches]
    drift_points = np.cumsum(batch_sizes)
    
    return X, None, {"drift_points": drift_points}

def load_gas_data(path):
    """
    Load full gas dataset (all batches combined)
    Returns:
        X: (N, F)
        y: labels (or None)
        meta: dict with drift points
    """

    X_all = []
    y_all = []

    # Assuming batches: batch1 → batch10
    for i in range(1, 11):
        X_batch, y_batch = load_gas_batch(path, batch_id=i)

        X_all.append(X_batch)
        y_all.append(y_batch)

    X = np.vstack(X_all)
    y = np.concatenate(y_all) if y_all[0] is not None else None

    # Ground truth drift points (between batches)
    meta = {
        "drift_points": np.cumsum([len(b) for b in X_all[:-1]])
    }

    return X, y, meta