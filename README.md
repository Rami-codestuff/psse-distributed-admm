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
```bash
python Optimize_Partitioning.py
