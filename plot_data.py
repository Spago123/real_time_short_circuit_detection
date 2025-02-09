import numpy as np
import matplotlib.pyplot as plt
import argparse

## change window size according to model
WINDOW_SIZE = 15

def plot_data(file):
    # Load the data
    true_outputs = np.load(f'true_predictions_{file}.npy')
    predicted_outputs = np.load(F'predicted_predictions_{file}.npy')

    # Create a time axis (sample numbers)
    sample_numbers = np.arange(len(true_outputs))

    # Plot the true and predicted outputs
    plt.figure(figsize=(12, 6))
    plt.plot(sample_numbers, true_outputs, 'b-', label='True Output', alpha=0.7)
    plt.plot(sample_numbers, predicted_outputs, 'r--', label='Predicted Output', alpha=0.7)

    # Highlight errors
    errors = np.where(true_outputs != predicted_outputs)[0]
    plt.scatter(errors, true_outputs[errors], color='black', label='Errors', zorder=5)

    # Add labels, title, and grid
    plt.xlabel('Sample Number')
    plt.ylabel('Fault Class')
    plt.title('True vs Predicted Outputs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust axis limits
    plt.xlim(0, len(true_outputs))
    plt.ylim(-0.5, 7.5)  # Adjust based on your class range

    # Save the plot
    plt.savefig(f'figures/true_vs_predicted_{file}.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("-f", "--file", type=str, default="model_window_size_10", help="Input file")

    args = parser.parse_args()

    print(f"File: {args.file}")
    plot_data(args.file)