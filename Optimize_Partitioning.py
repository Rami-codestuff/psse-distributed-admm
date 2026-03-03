import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from networkx.algorithms.community import modularity

# --- CONFIGURATION ---
from Solver_test1 import PowerFlowSolver

RAW_FILENAME = "IEEE 30 Bus.RAW"

# Range for the Elbow Plot
MIN_K = 2
MAX_K = 20

# The Partition we actually want to use
SELECTED_K = 4


# ---------------------

def get_spectral_partition(G, k):
    """Performs Spectral Clustering on Graph G into k clusters."""
    adj_matrix = nx.to_numpy_array(G)
    sc = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=100,
                            assign_labels='discretize', random_state=42)
    labels = sc.fit_predict(adj_matrix)

    # Convert to sets for modularity calc
    nodes = list(G.nodes())
    communities = []
    for i in range(k):
        comm = {nodes[n] for n in range(len(labels)) if labels[n] == i}
        if comm: communities.append(comm)
    return communities, labels


def plot_elbow_curve(k_values, scores):
    """Generates the specific 'Elbow Plot' for your thesis."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, marker='o', linestyle='-', color='b', linewidth=2)

    # Highlight the selected K
    selected_score = scores[k_values.index(SELECTED_K)]
    plt.plot(SELECTED_K, selected_score, marker='o', markersize=12, color='r', label=f'Selected (k={SELECTED_K})')

    plt.title("Modularity Score vs. Number of Partitions (IEEE 30 Bus)", fontsize=14)
    plt.xlabel("Number of Partitions (k)", fontsize=12)
    plt.ylabel("Modularity Score (Q)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    filename = "partition_elbow_plot.png"
    plt.savefig(filename, dpi=300)
    print(f"Elbow plot saved to '{filename}'.")
    plt.close()


def save_partition_data(G, labels, k):
    """Saves the bus allocation to a text file."""
    filename = "partition_allocation.txt"
    nodes = list(G.nodes())
    with open(filename, "w") as f:
        f.write(f"# Partition Allocation for Parallel Solver\n")
        f.write(f"# Source: {RAW_FILENAME}\n")
        f.write(f"# K: {k}\n")
        f.write(f"BusID,ZoneID\n")
        for i, node in enumerate(nodes):
            f.write(f"{node},{labels[i]}\n")
    print(f"Partition data saved to '{filename}'.")


def visualize_partition(G, labels, k, score):
    """Generates the HD map for the selected K."""
    print(f"Generating HD Map for k={k}...")

    # Identify Cut Lines
    cut_lines = []
    node_cluster_map = {node: labels[i] for i, node in enumerate(G.nodes())}
    for u, v in G.edges():
        if node_cluster_map[u] != node_cluster_map[v]:
            cut_lines.append((u, v))

    # Plot
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=100)

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color=labels, cmap=plt.get_cmap('tab10'),
                           node_size=400, edgecolors='black')

    # Draw Internal Edges
    normal_edges = [e for e in G.edges() if e not in cut_lines and (e[1], e[0]) not in cut_lines]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray', alpha=0.3)

    # Draw Cut Lines
    nx.draw_networkx_edges(G, pos, edgelist=cut_lines, edge_color='red', style='dashed', width=3)

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    plt.title(f"Final Grid Partitioning (k={k}, Modularity={score:.4f})", fontsize=22)
    plt.axis('off')

    output_file = f"partition_map_k{k}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Map saved to '{output_file}'.")
    plt.close()


def main():
    # 1. Load Data
    print(f"Loading Grid: {RAW_FILENAME}...")
    solver = PowerFlowSolver(RAW_FILENAME)
    solver.parse_raw_file()
    solver.build_ybus()

    # 2. Build Weighted Graph
    G = nx.Graph()
    G.add_nodes_from(solver.bus_df.index)
    for br in solver.branch_data:
        x_val = abs(br['x']) if abs(br['x']) > 1e-6 else 1e-6
        G.add_edge(br['from'], br['to'], weight=1.0 / x_val)

    print(f"Graph Built: {G.number_of_nodes()} Nodes. Running Analysis...")

    # 3. Run Sweep (k=2 to 20)
    k_values = []
    scores = []

    # Variables to store our SELECTED partition (k=4)
    selected_labels = None
    selected_score = 0

    print(f"{'K':<5} | {'Score':<10}")
    print("-" * 20)

    for k in range(MIN_K, MAX_K + 1):
        communities, labels = get_spectral_partition(G, k)
        try:
            score = modularity(G, communities, weight='weight')
        except:
            score = 0

        k_values.append(k)
        scores.append(score)
        print(f"{k:<5} | {score:<10.5f}")

        if k == SELECTED_K:
            selected_labels = labels
            selected_score = score

    # 4. Generate Outputs
    print("\n--- Generating Thesis Artifacts ---")
    plot_elbow_curve(k_values, scores)
    save_partition_data(G, selected_labels, SELECTED_K)
    visualize_partition(G, selected_labels, SELECTED_K, selected_score)
    print("\nDONE. You are ready for Parallel Solving.")


if __name__ == "__main__":
    main()