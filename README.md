# Constraint-Enforced Logistics Optimization (Toy Problem)

A proof-of-concept repository applying custom loss functions in PyTorch to enforce capacity and demand constraints in a multi-node supply chain network. 

This project was developed as a personal exploration into Operations Research (OR) and Mathematical Programming, aimed at my application to the **Department of Industrial Engineering at UNIST**.

## Motivation
Inspired by research on distributed optimization and real-time network operations, this notebook demonstrates a simplified Machine Learning approach to solving constrained logistics problems. Instead of relying purely on standard linear solvers, it uses gradient descent and penalty heuristics to enforce strict physical limits (hub capacities) and service requirements (node demands).

## Repository Contents
- `Logistics_Optimization_PoC.ipynb`: The main notebook containing mathematical formulations (Objective Function & Constraints), PyTorch custom loss implementation, and network visualization.
- `hubs.csv`: Synthetic data representing 5 Distribution Hubs with capacity limits.
- `nodes.csv`: Synthetic data representing 20 Delivery Nodes with specific demands.

## Optimization Logic
The model avoids "trivial solutions" (e.g., zero transport flow) by employing a two-part penalty constraint alongside the transport cost minimization:
1. **Capacity Penalty:** Penalizes the model if outflow exceeds maximum storage.
2. **Demand Penalty:** Penalizes the model if inflow fails to meet customer demand.
