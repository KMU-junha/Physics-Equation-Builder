import pandas as pd
import numpy as np
import os

def generate_laplace_data():
    """
    라플라스 변환 매핑 데이터를 생성하여 CSV로 저장합니다.
    입력 함수: f(t) = t
    출력 변환: F(s) = 1 / s^2
    """
    # 임의의 s 범위에 대한 100개의 데이터 샘플 생성
    s = np.linspace(1.0, 10.0, 100)
    
    # 식: F(s) = 1 / s^2
    F_s_true = 1.0 / (s ** 2)
    
    # 현실성을 위해 오차 추가
    np.random.seed(42)
    noise_level = 0.05
    F_s_noisy = F_s_true + np.random.normal(0, noise_level * np.mean(F_s_true), size=len(s))
    
    # F(s)는 s>0에서 양수여야 하므로 0 이상의 값 보장
    F_s_noisy = np.maximum(F_s_noisy, 0.0001)
    
    # 모델 입력 형태와 맞추기 위해 t(더미) 변수를 0으로 둔다.
    # 우리의 SINDy/gplearn 모델은 (d, t) -> v 형태의 두 개의 피처 입력을 기대하므로,
    # 여기서는 X0 = s, X1 = 0 으로 구성하여 F(s)를 예측시킨다.
    df = pd.DataFrame({
        'd': s,      # 실제로는 s 공간
        't': np.zeros_like(s),      # 더미 변수
        'v': F_s_noisy   # 실제로는 F(s) 결과값
    })
    
    # 저장 경로 지정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    save_path = os.path.join(data_dir, 'laplace_data.csv')
    df.to_csv(save_path, index=False)
    
    print(f"새로운 실험 데이터가 생성되었습니다: {save_path}")
    print("목표: gplearn이 입력 s에 대해 결과값이 1/(s^2)의 역제곱 법칙을 찾아내야 합니다.")

if __name__ == "__main__":
    generate_laplace_data()
