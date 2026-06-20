import pandas as pd
import sympy
from sympy import Symbol, sin, cos, exp
import os

def generate_taylor_poly_data():
    """
    [테일러 2차 근사 — 다항식 변조 (Polynomial Modulation)]
    Target operator: output = f_t * (1 + a*f_t + b*f_t²)

    물리적 의미:
      - f_t  : 임의의 물리량 (변위, 진폭, 전압 등)
      - a, b : 테일러 계수 (1차, 2차 보정 항)
      - 비선형 응답 모델링에서 등장하는 다항식 커널 구조

    엔진이 찾아야 할 커널:
      - 템플릿: f_t * Unknown_Term
      - 커널:   1 + a*ph + b*ph²   (Add/Mul/Pow(2) 조합, 깊이 3)

    쇼트컷 치환 흐름:
      input=t       → subs(t, ph)      → ph*(1 + a*ph + b*ph²)  ✓
      input=2t      → subs(2t→ph, 4t²→ph², 8t³→ph³) → ph*(1 + a*ph + b*ph²) ✓
      input=sin(ωt) → subs(sin→ph, sin²→ph², sin³→ph³) → ph*(1 + ...) ✓
    """
    t     = Symbol('t')
    omega = Symbol('omega')
    a     = Symbol('a')
    b     = Symbol('b')
    k     = Symbol('k')

    inputs = [
        t,
        2 * t,
        sin(omega * t),
        exp(-k * t),
        t + 1,
    ]

    data = []
    for f_t in inputs:
        target_expr = f_t * (1 + a * f_t + b * f_t**2)
        data.append({
            'input':  str(f_t),
            'output': str(target_expr),   # expand 금지: 인수 구조 보존
            'roc':    ''
        })

    df = pd.DataFrame(data)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'taylor_poly_symbolic.csv')
    df.to_csv(csv_path, index=False)
    print(f"[생성 완료] {csv_path}")
    print(df.to_string(index=False))

if __name__ == '__main__':
    generate_taylor_poly_data()
