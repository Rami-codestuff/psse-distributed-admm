# Distributed PSS/E ADMM Solver using Spectral Clustering

This repository contains the complete pipeline for geographically partitioning a power grid and solving the power flow using a distributed Alternating Direction Method of Multipliers (ADMM) architecture. 

It orchestrates multiple independent PSS®E (Power System Simulator for Engineering) engines via Python `multiprocessing` to negotiate boundary voltages without requiring a centralized matrix solver. This architecture is designed to address the $O(N^3)$ computational scaling bottlenecks of centralized Newton-Raphson solvers on massive grids.

## 🧠 The Partitioning Logic: Spectral Clustering
To ensure the distributed solver is as efficient as possible, the grid is not cut arbitrarily. The algorithm minimizes the number of cut lines (Tie-Lines) to reduce communication overhead, while keeping the internal zones electrically cohesive.

This is achieved mathematically using **Spectral Clustering**:
1. **Graph Representation:** The grid is modeled as an Adjacency Matrix where Buses are Nodes and Transmission Lines are Edges.
2. **The Laplacian Matrix:** We calculate the Graph Laplacian (L = D - A), which mathematically maps the connectivity of the entire grid.
3. **Eigenvalue Decomposition:** We extract the eigenvectors corresponding to the smallest non-zero eigenvalues of the Laplacian. These vectors expose the natural "fault lines" or bottlenecks in the grid topology.
4. **K-Means Clustering:** We apply K-Means to these eigenvectors to group the buses into optimal geographic zones (e.g., k=4). 
5. **Output:** This generates a `partition_allocation.txt` map, guiding the network splitter.

## ⚙️ Prerequisites
* **PSS®E 36.5** installed and licensed.
* Python 3.14 (or compatible PSS®E Python environment) properly configured with the `psspy` API.
* Required Python libraries: `numpy`, `scikit-learn`, `networkx`, `psutil`

## 🚀 Steps to Replicate

Ensure the master `IEEE 30 bus.RAW` file is in the root directory, then run the scripts in the following exact order using the PSS®E Command Prompt:

### Step 1: Generate the Partitions
`python Optimize_Partitioning.py`
* **What it does:** Runs the Spectral Clustering algorithm on the network.
* **Output:** Generates the `partition_allocation.txt` map and saves visual plots of the optimal cuts.

### Step 2: Perform Network Surgery
`python PSSE_Splitter.py`
* **What it does:** Uses PSS/E to read the allocation map, sever the physical tie-lines, and inject "Virtual Slack" (Dummy Generators) at the boundaries to maintain electrical stability for the isolated zones.
* **Output:** Generates fully independent sub-grid files (`Zone_0.raw`, `Zone_1.raw`, etc.).

### Step 3: Run the True Multi-Core ADMM Solver
`python PSSE_Parallel_ADMM.py`
* **What it does:** Spawns separate Python/PSS®E instances across your CPU cores. Each core solves its local `Zone_X.raw` matrix using Newton-Raphson. The master script calculates the boundary mismatches and updates the Dummy Generators using an Alpha learning rate until the boundary voltages completely synchronize.
* **Output:** * Live terminal feed showing boundary errors dropping to zero.
  * Validation table comparing the distributed ADMM result against a centralized Newton-Raphson solver.
  * Automatically saves `ADMM_Final_Results.txt` containing the fully solved network voltage profile.
