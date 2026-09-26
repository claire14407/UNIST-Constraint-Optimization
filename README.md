# Constraint-Enforced Logistics Network Optimization via PyTorch

## A Proof-of-Concept (PoC) applying gradient-based penalty relaxation to multi-node supply chain allocation problems.

This repository presents a minimal academic simulation of a constrained logistics network (5 Hubs, 20 Demand Nodes) developed for my undergraduate application to the **Department of Industrial Engineering at UNIST**.

It bridges **predictive analytics (Business Analytics)** and **prescriptive operations research (OR)** by demonstrating how neural gradient descent can respect explicit physical bounds without relying exclusively on traditional MILP solvers.

### Key Features & Technical Highlights

* **Mathematical Formulation:** Explicit definition of transport cost minimization subject to hub capacity limits and demand requirements.
* **Custom PyTorch Penalty Loss:** Implements continuous `torch.relu` loss functions to penalize capacity overflow and under-delivery during backpropagation.
* **Trivial Solution Avoidance:** Overcomes the trivial $X=0$ zero-flow edge case by enforcing dual-sided penalty bounds (Outflow vs. Inflow).
* **Synthetic Dataset Pipeline:** Includes standalone Python scripts to generate spatial topology data (`hubs.csv`, `nodes.csv`).
* **Network Visualization:** Integrated `matplotlib` scripts rendering hub capacity distributions and node demand coordinates.

### Mathematical Model

#### Objective Function (Cost Minimization)
$$\min \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij}$$

#### Operational Constraints
* **Hub Capacity:** $\sum_{j=1}^{n} x_{ij} \leq C_i \quad \forall i \in \{1, \dots, m\}$
* **Node Demand:** $\sum_{i=1}^{m} x_{ij} \geq D_j \quad \forall j \in \{1, \dots, n\}$

### Repository Contents

* `Logistics_Optimization_PoC.ipynb` : Main Jupyter Notebook containing formulations, code implementation, training loops, and plots.
* `hubs.csv` : Dataset containing coordinates and capacity bounds for 5 distribution hubs.
* `nodes.csv` : Dataset containing coordinates and demand requirements for 20 delivery nodes.

### Future Academic Goals at UNIST

While effective for this toy problem, simple penalty methods can face convergence challenges in high-dimensional networks. At UNIST, I aim to explore advanced exact and dynamic techniques, such as **Lagrangian Relaxation** and **ADMM**, to scale optimization models for complex industrial systems.
