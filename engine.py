import pandas as pd
import numpy as np

def generate_data():
    # 10 Global Suppliers
    suppliers = [f"Supplier_{i}" for i in range(1, 11)]
    
    data = {
        'Supplier': suppliers,
        'Unit_Cost': np.random.uniform(10, 50, 10),
        'Lead_Time': np.random.uniform(5, 30, 10),
        'Carbon_Footprint': np.random.uniform(1, 10, 10),
        'Reliability': np.random.uniform(0.85, 0.99, 10), # For Monte Carlo later
        'Capacity': np.random.randint(500, 2000, 10)
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_data()
    print("--- Initial Supplier Data ---")
    print(df.head())