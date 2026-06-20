import os
import pandas as pd
import numpy as np

def generate_drag_data():
    """
    공기저항(선형 감쇠)을 받는 물체의 속도-거리 데이터 생성 스크립트
    위치: d(t) = (v0/k) * (1 - e^{-kt})
    속도: v(t) = v0 * e^{-kt}
    
    물리적 관계: v = v0 - k*d 의 선형 관계를 가지므로 
    SINDy가 정확하게 찾아낼 수 있는지를 테스트하기 아주 좋은 데이터셋입니다.
    """
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "drag_motion_data.csv")
    
    t = np.linspace(0, 5, 200) # 0초부터 5초까지 200개 데이터
    v0 = 10.0 # 초기 속도 (m/s)
    k = 2.0   # 공기저항/감쇠 계수
    
    d = (v0 / k) * (1 - np.exp(-k * t))
    v = v0 * np.exp(-k * t)
    
    df = pd.DataFrame({"t": t, "d": d, "v": v})
    df.to_csv(out_path, index=False)
    
    print(f"🧹 [신규 데이터셋] 공기저항 감쇠 운동 데이터 생성 완료: {out_path}")
    print(df.head())

if __name__ == "__main__":
    generate_drag_data()
