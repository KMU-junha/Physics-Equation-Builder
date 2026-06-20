import os
import pandas as pd
import numpy as np

def generate_constant_accel_data():
    """
    Generates a numerical dataset for constant acceleration kinematics.
    Equation: d = 0.5 * a * t^2 + v0 * t + d0
              v = a * t + v0
    """
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "constant_accel_data.csv")
    
    t = np.linspace(0, 10, 200) # 200 data points from 0 to 10s
    a = 9.8                     # Constant acceleration (m/s^2)
    v0 = 0.0                    # Initial velocity (m/s)
    d0 = 0.0                    # Initial position (m)
    
    d = 0.5 * a * (t ** 2) + v0 * t + d0
    v = a * t + v0
    
    # Add a slight realistic noise to position and velocity if needed, but let's keep it clean for base testing
    # v += np.random.normal(0, 0.01, size=v.shape)
    
    df = pd.DataFrame({"t": t, "d": d, "v": v})
    df.to_csv(out_path, index=False)
    
    print(f"✅ Generated Constant Acceleration Dataset: {out_path}")
    print(f"   Shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    generate_constant_accel_data()
