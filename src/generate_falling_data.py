import pandas as pd
import numpy as np
import os

def generate_falling_object_data():
    """
    등가속도 운동 (자유낙하) 데이터를 생성하여 CSV로 저장합니다.
    d = 0.5 * g * t^2   (g = 9.8 m/s^2)
    v = g * t
    """
    # 0.1초 단위로 10초 동안의 데이터를 생성 (총 100개 샘플)
    t = np.linspace(0.1, 10.0, 100)
    
    # 중력가속도
    g = 9.8 
    
    # 실제 수식 (노이즈 없는 이상적인 값)
    d_true = 0.5 * g * (t ** 2)
    v_true = g * t
    
    # 관측값에 약간의 랜덤 노이즈(오차) 추가 (현실성 부여)
    np.random.seed(42)
    noise_level = 0.05
    d_noisy = d_true + np.random.normal(0, noise_level * np.mean(d_true), size=len(t))
    v_noisy = v_true + np.random.normal(0, noise_level * np.mean(v_true), size=len(t))
    
    # 음수는 0으로 절사
    d_noisy = np.maximum(d_noisy, 0)
    v_noisy = np.maximum(v_noisy, 0)
    
    # DataFrame 생성
    df = pd.DataFrame({
        'd': d_noisy,
        't': t,
        'v': v_noisy
    })
    
    # 저장 경로 지정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    save_path = os.path.join(data_dir, 'falling_object_data.csv')
    df.to_csv(save_path, index=False)
    
    print(f"새로운 실험 데이터가 생성되었습니다: {save_path}")
    print("수식 특징: 가속도(g=9.8)가 작용하여 속도 v와 거리 d가 시간에 따라 변합니다.")

if __name__ == "__main__":
    generate_falling_object_data()
