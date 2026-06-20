import pandas as pd
import os
import sympy
from sympy import Symbol, exp, sqrt, pi, I, oo, Heaviside, Abs

def generate_fourier_omega_data():
    """
    [푸리에 변환 (Fourier Transform - Omega Domain)]
    정의: F(w) = Integral(f(t) * exp(-I*w*t), (t, -oo, oo))
    """
    # 데이터 정의 (기호식 문자열)
    data = [
        {
            "input": "exp(-a*t**2)", 
            "output": "sqrt(pi/a) * exp(-w**2/(4*a))",
            "roc": "re(a) > 0"
        },
        {
            "input": "exp(-a*t) * Heaviside(t)", 
            "output": "1/(a + I*w)",
            "roc": "re(a) > 0"
        },
        {
            "input": "exp(-a*Abs(t))", 
            "output": "2*a/(a**2 + w**2)",
            "roc": "re(a) > 0"
        },
        {
            "input": "1/(a**2 + t**2)", 
            "output": "(pi/a) * exp(-a*Abs(w))",
            "roc": "re(a) > 0"
        },
        {
            "input": "Heaviside(t + 1/2) - Heaviside(t - 1/2)", 
            "output": "sin(w/2) / (w/2)",
            "roc": "True"
        }
    ]
    
    df = pd.DataFrame(data)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "fourier_omega_symbolic.csv")
    df.to_csv(csv_path, index=False)
    
    print("="*50)
    print(f" 생성된 CSV: {csv_path}")
    print("-"*50)
    print(df.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    generate_fourier_omega_data()
