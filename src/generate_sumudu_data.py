import pandas as pd
import os
import sympy
from sympy import Symbol, exp, sin, cos

def generate_sumudu_data():
    """
    [수무두 변환 (Sumudu Transform)]
    정의: G(u) = (1/u) * Integral(f(t) * exp(-t/u), (t, 0, oo))
    
    특징: 
      - f(t) = 1    -> G(u) = 1
      - f(t) = t    -> G(u) = u
      - f(t) = t^2  -> G(u) = 2*u^2
      - f(t) = e^{at} -> G(u) = 1/(1 - a*u)
      - f(t) = sin(at) -> G(u) = a*u / (1 + a^2*u^2)
    """
    data = [
        {"input": "1", "output": "1", "roc": "u > 0"},
        {"input": "t", "output": "u", "roc": "u > 0"},
        {"input": "t**2", "output": "2*u**2", "roc": "u > 0"},
        {"input": "exp(a*t)", "output": "1/(1 - a*u)", "roc": "u < 1/a"},
        {"input": "sin(a*t)", "output": "a*u / (1 + a**2 * u**2)", "roc": "u > 0"},
    ]
    
    df = pd.DataFrame(data)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "sumudu_symbolic.csv")
    df.to_csv(csv_path, index=False)
    print(f"[생성 완료] {csv_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    generate_sumudu_data()
