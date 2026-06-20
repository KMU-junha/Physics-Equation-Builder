import pandas as pd
import numpy as np
import os
import random

def generate_multi_laplace_data():
    """
    여러 가지 기저 함수들의 라플라스 변환 매핑을 무작위로 조합하여 10개의 데이터셋으로 저장합니다.
    """
    s = np.linspace(2.0, 15.0, 150) # s > a 조건을 만족하기 위해 s 범위를 2부터 시작
    np.random.seed(42)
    random.seed(42)
    noise_level = 0.005 # 노이즈를 살짝 줄여서 gplearn이 조합을 찾기 쉽게 도움

    base_funcs = [
        {"desc": "1/s", "val": 1.0 / s},
        {"desc": "1/s^2", "val": 1.0 / (s ** 2)},
        {"desc": "1/(s-1)", "val": 1.0 / (s - 1.0)},
        {"desc": "2/(s^2+4)", "val": 2.0 / (s**2 + 4.0)}
    ]

    datasets = {}

    # 10개의 무작위 선형 결합 생성
    for i in range(1, 11):
        # 2개 또는 3개의 기저 함수를 무작위로 선택
        num_funcs = random.choice([2, 3])
        chosen = random.sample(base_funcs, num_funcs)
        
        combined_val = np.zeros_like(s)
        desc_parts = []
        for func in chosen:
            # 1~5 사이의 정수 계수 곱하기
            coef = float(random.randint(1, 5))
            combined_val += coef * func["val"]
            
            if coef == 1.0:
                desc_parts.append(f"{func['desc']}")
            else:
                desc_parts.append(f"{int(coef)}*({func['desc']})")
        
        desc = " + ".join(desc_parts)
        datasets[f"laplace_combo_{i}"] = {
            "desc": desc,
            "F_s": combined_val
        }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    for name, data in datasets.items():
        F_s_true = data["F_s"]
        # 약간의 관측 노이즈 추가
        F_s_noisy = F_s_true + np.random.normal(0, noise_level * np.mean(F_s_true), size=len(s))
        F_s_noisy = np.maximum(F_s_noisy, 0.00001)

        # X0=d (s에 대응), X1=t (더미 0)
        df = pd.DataFrame({
            'd': s,      
            't': np.zeros_like(s),      
            'v': F_s_noisy   
        })
        
        save_path = os.path.join(data_dir, f'{name}.csv')
        df.to_csv(save_path, index=False)
        print(f"생성 완료: {name}.csv ({data['desc']})")

if __name__ == "__main__":
    generate_multi_laplace_data()
