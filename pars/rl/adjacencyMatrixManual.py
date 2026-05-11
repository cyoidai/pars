import numpy as np

# ==========================================
# CONFIGURATION: EDIT YOUR GOALS AND MAP HERE
# ==========================================

# 1. Choose the cities the AI MUST visit
MANUAL_GOALS = [2, 5, 8]

# 2. Define your 10x10 Adjacency Matrix in [[],[],...] form
# 0.0 means NO connection. Any other number is the travel cost.
matrix_list = [
    [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0], # Row 0
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Row 1
    [0.4, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0], # Row 2
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Row 3
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Row 4
    [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0], # Row 5
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Row 6
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Row 7
    [0.9, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0], # Row 8
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Row 9
]

# ==========================================
# LOGIC: CONVERSION AND SAVING
# ==========================================

def get_manual_data():
    # Converts the list format to a numpy array for the AI
    m = np.array(matrix_list, dtype=np.float32)
    return m, MANUAL_GOALS

if __name__ == "__main__":
    m, goals = get_manual_data()
    np.save("train_map.npy", m)
    print("Manual matrix saved to train_map.npy")
    print(f"Manual goals defined as: {goals}")