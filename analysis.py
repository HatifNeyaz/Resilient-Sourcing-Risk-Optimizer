import pulp
import pandas as pd

class DEASolver:
    def __init__(self, df, inputs, outputs):
        self.df = df
        self.inputs = inputs
        self.outputs = outputs
        self.suppliers = df.index.tolist()

    def solve_efficiency(self, target_idx):
        # Create the LP problem
        prob = pulp.LpProblem(f"DEA_{target_idx}", pulp.LpMaximize)
        
        # Decision Variables: weights for inputs (v) and outputs (u)
        v = pulp.LpVariable.dicts("v", self.inputs, lowBound=0.0001)
        u = pulp.LpVariable.dicts("u", self.outputs, lowBound=0.0001)
        
        # Objective: Maximize weighted output of the target supplier
        prob += pulp.lpSum([u[o] * self.df.loc[target_idx, o] for o in self.outputs])
        
        # Constraint 1: Weighted input of target supplier must be 1 (Normalization)
        prob += pulp.lpSum([v[i] * self.df.loc[target_idx, i] for i in self.inputs]) == 1
        
        # Constraint 2: For all suppliers, weighted output <= weighted input
        for idx in self.suppliers:
            out_sum = pulp.lpSum([u[o] * self.df.loc[idx, o] for o in self.outputs])
            in_sum = pulp.lpSum([v[i] * self.df.loc[idx, i] for i in self.inputs])
            prob += out_sum <= in_sum
            
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return pulp.value(prob.objective)

    def get_all_scores(self):
        scores = []
        for i in self.df.index:
            scores.append(self.solve_efficiency(i))
        return scores
    




# ============== ANP ==================

import numpy as np

def calculate_ahp_weights(matrix):
    """
    Calculates weights from a pairwise comparison matrix.
    The matrix should be square (n x n).
    """
    matrix = np.array(matrix)
    # Eigenvector method (approximate)
    column_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / column_sums
    weights = normalized_matrix.mean(axis=1)
    
    # Consistency Check (Crucial for showing off your Math depth)
    n = matrix.shape[0]
    eig_val, _ = np.linalg.eig(matrix)
    max_eig = max(eig_val).real
    ci = (max_eig - n) / (n - 1)
    ri = {3: 0.58, 4: 0.90, 5: 1.12} # Random Index for consistency
    cr = ci / ri.get(n, 1.0)
    
    return weights, cr

# Phase 2: Allocation Engine

import pulp

class AllocationSolver:
    def __init__(self, df, total_demand):
        self.df = df
        self.total_demand = total_demand
        self.suppliers = df.index.tolist()

    def solve_allocation(self, lambda_weight=10.0):
        # Define the LP Problem
        prob = pulp.LpProblem("Sourcing_Allocation", pulp.LpMinimize)
        
        # Decision Variables: Quantity to order from each supplier
        # x[i] is the quantity for supplier i
        x = pulp.LpVariable.dicts("OrderQty", self.suppliers, lowBound=0, cat='Continuous')
        
        # Objective Function: Min (Total Cost) - (Efficiency Bonus)
        # We subtract the efficiency bonus because we are MINIMIZING
        cost_term = pulp.lpSum([self.df.loc[i, 'Unit_Cost'] * x[i] for i in self.suppliers])
        eff_term = pulp.lpSum([self.df.loc[i, 'Efficiency_Score'] * x[i] for i in self.suppliers])
        
        prob += cost_term - (lambda_weight * eff_term)
        
        # Constraints
        # 1. Meet Total Demand
        prob += pulp.lpSum([x[i] for i in self.suppliers]) == self.total_demand
        
        # 2. Stay within Supplier Capacity
        for i in self.suppliers:
            prob += x[i] <= self.df.loc[i, 'Capacity']
            
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract Results
        allocations = {i: pulp.value(x[i]) for i in self.suppliers}
        return allocations, pulp.LpStatus[prob.status]
    

# Monte Carlo Simulation for Risk Assessment

import numpy as np

class SimulationEngine:
    def __init__(self, df, allocations, iterations=1000):
        self.df = df
        self.allocations = allocations # Dictionary from Phase 2
        self.iterations = iterations

    def run_risk_simulation(self):
        # We only care about suppliers we actually ordered from
        active_suppliers = [s for s, qty in self.allocations.items() if qty > 0]
        
        results = []
        
        for _ in range(self.iterations):
            total_delivered = 0
            for s in active_suppliers:
                qty = self.allocations[s]
                reliability = self.df.loc[s, 'Reliability']
                
                # Binomial distribution: Did the 'shipment' succeed?
                # For high-volume, we can use a Normal distribution around the mean
                actual_qty = qty * np.random.binomial(1, reliability)
                
                # Alternative: Continuous disruption (Lead time delay impact)
                # actual_qty = qty * np.random.uniform(reliability, 1.0) 
                
                total_delivered += actual_qty
            
            results.append(total_delivered)
            
        return np.array(results)
    
