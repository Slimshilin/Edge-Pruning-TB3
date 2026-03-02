import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

IDEAL_RANGE = (0.975, 0.99)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize evaluation results from CSV.")
    parser.add_argument("--eval_results_csv_path", type=str, required=True, help="Path to the evaluation results CSV file.")
    parser.add_argument("--save_to_file", type=str, required=True, help="Path to save the output PNG file.")
    parser.add_argument("--edge_sparsity_lower_bound", type=float, default=0.0, help="Lower bound for edge sparsity.")
    parser.add_argument("--figure_title", type=str, default="Evaluation Results", help="Title for the entire figure.")
    return parser.parse_args()

def load_data(csv_path, lower_bound):
    df = pd.read_csv(csv_path)
    baseline = df[df['es'] == 0].iloc[0]
    filtered_df = df[(df['Overall Edge Sparsity'] >= lower_bound) & (df['es'] != 0)]
    return filtered_df, baseline

def plot_metric(ax, df, metric, baseline, color, marker):
    sparsity = df['Overall Edge Sparsity']
    in_range = (sparsity >= IDEAL_RANGE[0]) & (sparsity <= IDEAL_RANGE[1])

    # Out-of-range points: diluted
    if (~in_range).any():
        ax.scatter(sparsity[~in_range], df.loc[~in_range, metric],
                   color=color, marker=marker, s=50, alpha=0.2, zorder=2)
    # In-range points: full opacity
    if in_range.any():
        ax.scatter(sparsity[in_range], df.loc[in_range, metric],
                   color=color, marker=marker, s=50, alpha=1.0, zorder=3)

    ax.axhline(y=baseline[metric], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Overall Edge Sparsity', fontsize=12, labelpad=3)
    ax.set_ylabel(metric, fontsize=12, labelpad=3)

    ax.set_facecolor('#f5f5f5')
    ax.grid(True, color='white', linestyle='-', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=10)

def visualize_results(df, baseline, save_path, figure_title):
    metrics = ['Accuracy', 'Logit difference', 'KL Divergence', 'Exact Match']
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
    markers = ['o', 's', '^', 'D']

    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5))
    fig.suptitle(figure_title, fontsize=14, fontweight='bold', y=0.99)

    for i, (metric, color, marker) in enumerate(zip(metrics, colors, markers)):
        ax = axs[i // 2, i % 2]
        plot_metric(ax, df, metric, baseline, color, marker)

    custom_lines = [Line2D([0], [0], color=c, marker=m, linestyle='None', markersize=5)
                    for c, m in zip(colors, markers)]
    custom_lines.append(Line2D([0], [0], color='red', linestyle='--', lw=1.5))

    fig.legend(custom_lines, metrics + ['Baseline (no pruning)'],
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96], h_pad=1.5, w_pad=1.5)
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    print(f"Visualization saved to {save_path}")

def main():
    args = parse_arguments()
    df, baseline = load_data(args.eval_results_csv_path, args.edge_sparsity_lower_bound)
    visualize_results(df, baseline, args.save_to_file, args.figure_title)

if __name__ == "__main__":
    main()
