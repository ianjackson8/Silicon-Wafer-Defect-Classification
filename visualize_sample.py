import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the DataFrame from the pickle file
df = pd.read_pickle('Clean_Train_WM811K.pkl')

def wafer_area(wafer):
    """
    Computes the area (rows * columns) of a wafer map.
    Converts the wafer map to a NumPy array for a reliable shape calculation.
    Returns 0 if the wafer map is empty.
    """
    arr = np.array(wafer)
    if arr.size == 0:
        return 0
    return arr.shape[0] * arr.shape[1]

if False:
    # Compute area for each waferMap entry and add it as a new column to the DataFrame.
    df['area'] = df['waferMap'].apply(wafer_area)
    # print(df.head())  # Display the first few rows of the DataFrame for debugging.

    # Sort the DataFrame in descending order by area (largest wafer maps first)
    df_sorted = df.sort_values(by='area', ascending=False).reset_index(drop=True)

    # Let the user choose which wafer to plot based on sorted index.
    # idx=0 will be the largest wafer map, idx=1 the second largest, etc.
    idx_to_plot = 19  # Change this to plot a different wafer in sorted order.

    # Check that the provided index is within bounds.
    if idx_to_plot < 0 or idx_to_plot >= len(df_sorted):
        raise IndexError("Chosen index is out of range.")

    # Retrieve the chosen wafer map, convert to NumPy array for plotting.
    chosen_wafer = np.array(df_sorted.loc[idx_to_plot, 'waferMap'])

    # Print the shape and area information for debugging.
    print(f"Plotting wafer map at sorted index {idx_to_plot} with dimensions {chosen_wafer.shape} and area {df_sorted.loc[idx_to_plot, 'area']}")

    # Plot the chosen wafer map.
    plt.figure(figsize=(8, 6))
    plt.imshow(chosen_wafer, cmap='viridis', interpolation='none')
    # plt.colorbar()
    # plt.title(f"Wafer Map at Sorted Index {idx_to_plot} (Area: {df_sorted.loc[idx_to_plot, 'area']})")
    plt.title(f"Sample Wafer Map (Type: {df_sorted.loc[idx_to_plot, 'failureType']})")
    plt.show()

unique_failure_types = df['failureType'].unique()
num_slots = 8
selected_types = unique_failure_types[:num_slots]

# Create a dictionary mapping each selected failure type to one sample wafer map.
samples = {}
for ftype in selected_types:
    # Select the first sample for this failure type
    sample_row = df[df['failureType'] == ftype].iloc[0] #0
    samples[ftype] = sample_row['waferMap']

# Create a 4x2 grid of subplots.
fig, axes = plt.subplots(4, 2, figsize=(12, 16))
axes = axes.flatten()

# Plot a wafer map sample for each selected failure type.
for i, (failure, wafer) in enumerate(samples.items()):
    wafer_array = np.array(wafer)  # Convert the 2D list to a NumPy array
    axes[i].imshow(wafer_array, cmap='viridis', interpolation='none')
    axes[i].set_title(failure)
    axes[i].axis('off')  # Hide axes ticks and labels

# If there are any empty subplot slots, hide them.
for j in range(len(samples), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('sample_wafer_maps.png')