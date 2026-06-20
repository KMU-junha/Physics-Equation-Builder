import pandas as pd
import sympy
from sympy import Symbol, exp, sin, cos, log, sqrt
import os

def generate_algebraic_data():
    """
    [로렌츠 프로파일 (Lorentzian Profile)]
    Target operator: output = (a * f_t) / (f_t**2 + b**2)

    f_t가 분자에는 1차, 분모에는 2차로 동시에 등장하는 비선형 분수 구조.
    피크 너비(b)와 진폭(a)을 가진 공명/분산 형태로,
    광학 분광학, 전자기학 공명, 유체역학 등 다양한 물리 문제에서 등장함.
    
    [수정] omega, k 같은 심볼릭 파라미터 대신 숫자 상수 사용.
    → 자유 변수를 {a, b, t}로 최소화하여 GA 탐색 공간을 집중시킴.
    (심볼릭 파라미터가 있으면 GA가 미분형 탐색으로 오분류될 수 있음)
    """
    t = Symbol('t')
    a = Symbol('a')
    b = Symbol('b')

    # 다양한 입력 함수 형태 — 심볼릭 파라미터 없이 숫자 상수만 사용
    inputs = [
        t,
        2*t,
        sin(2*t),       # omega=2 (숫자)
        exp(-t),        # k=1 (숫자)
        t + 1,
        t**2,           # 다항식 형태 추가
        cos(t),         # 코사인 형태 추가
    ]

    data = []
    for f_t in inputs:
        # Target: a * f_t / (f_t**2 + b**2)
        target_expr = (a * f_t) / (f_t**2 + b**2)
        data.append({
            'input': str(f_t),
            'output': str(sympy.simplify(target_expr)),
            'roc': ''
        })

    df = pd.DataFrame(data)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'algebraic_symbolic.csv')
    df.to_csv(csv_path, index=False)
    print(f"[생성 완료] {csv_path}")
    print(df.to_string(index=False))

if __name__ == '__main__':
    generate_algebraic_data()
