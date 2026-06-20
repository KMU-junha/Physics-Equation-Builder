import pandas as pd
import sympy
from sympy import Symbol
import os

def generate_acoustic_reflection_data():
    """
    [음향 전달 계수 (Acoustic Transmission Coefficient)]
    Target operator: output = f_t / (f_t + b)

    물리적 의미:
      - f_t : Z1 (입사 매질의 음향 임피던스)
      - b   : Z2 (투과 매질의 음향 임피던스)
      - T_normalized = Z1 / (Z1 + Z2)   [전달 비율]

    엔진이 찾아야 할 커널 구조:
      - 템플릿: f_t * Unknown_Term
      - 커널: Pow( Add(f_t, b), -1 )   ← 깊이 2짜리 단순 구조
      - 즉: f_t * 1/(f_t + b) = f_t/(f_t + b)

    반사 계수와의 관계:
      R = (b - f_t)/(b + f_t) = 1 - 2*T_normalized
    → 반사 계수는 템플릿 분해 시 f_t가 분자/분모 양쪽에 비선형으로 결합되어
      현재 유전자 탐색 공간에서 수렴이 극히 어려움.
    → 전달 계수로 재정의하면 커널이 Pow(Add(f_t, b), -1) 로 단순화됨.
    """
    t = Symbol('t')
    omega = Symbol('omega')
    a_sym = Symbol('a')
    b = Symbol('b')
    k = Symbol('k')

    inputs = [
        t,
        2 * t,
        a_sym * t,
        a_sym * sympy.exp(-k * t),
        t + a_sym,
    ]

    data = []
    for f_t in inputs:
        # R = (Z2 - Z1) / (Z1 + Z2) = (b - f_t) / (f_t + b)
        target_expr = (b - f_t) / (f_t + b)
        data.append({
            'input': str(f_t),
            'output': str(sympy.simplify(target_expr)),
            'roc': ''
        })

    df = pd.DataFrame(data)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'acoustic_reflection_symbolic.csv')
    df.to_csv(csv_path, index=False)
    print(f"[생성 완료] {csv_path}")
    print(df.to_string(index=False))

if __name__ == '__main__':
    generate_acoustic_reflection_data()
