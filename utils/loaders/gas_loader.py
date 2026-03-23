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