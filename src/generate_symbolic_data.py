import pandas as pd
import os

def create_symbolic_datasets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1. 라플라스 변환 기호 테이블 (강건한 메타-학습을 위한 포괄적 세트)
    laplace_data = [
        {"input": "1", "output": "1/s", "roc": "s > 0"},                  # 상수 단위계단 함수
        {"input": "t", "output": "1/s**2", "roc": "s > 0"},               # 램프 함수
        {"input": "t**2", "output": "2/s**3", "roc": "s > 0"},            # 파라볼라 함수
        {"input": "exp(-t)", "output": "1/(s+1)", "roc": "s > 0"},        # 지수 감쇠 (a=-1)
        {"input": "exp(-2*t)", "output": "1/(s+2)", "roc": "s > 0"},      # 지수 감쇠 (a=-2)
        {"input": "t*exp(-t)", "output": "1/(s+1)**2", "roc": "s > 0"},   # 시간-지수 혼합
        {"input": "sin(t)", "output": "1/(s**2 + 1)", "roc": "s > 0"},    # 사인 진동
        {"input": "cos(t)", "output": "s/(s**2 + 1)", "roc": "s > 0"},    # 코사인 진동
        {"input": "sin(2*t)", "output": "2/(s**2 + 4)", "roc": "s > 0"},  # 사인 고주파
    ]
    df_laplace = pd.DataFrame(laplace_data)
    df_laplace.to_csv(os.path.join(data_dir, "laplace_symbolic.csv"), index=False)
    print("생성 완료: laplace_symbolic.csv")

    # 2. 2차 방정식 근의 공식 (파라미터 -> 함수)
    # 입력 방정식 구조: a*x^2 + b*x + c = 0
    # 출력: 근의 공식
    quadratic_data = [
        {"input": "1", "output": "(-b + sqrt(b**2 - 4*a*c))/(2*a)"}, 
        # 단일 매핑만으로도 공식을 엮어낼 수 있는지 테스트하기 위함 (Meta-learning)
    ]
    df_quad = pd.DataFrame(quadratic_data)
    df_quad.to_csv(os.path.join(data_dir, "quadratic_symbolic.csv"), index=False)
    print("생성 완료: quadratic_symbolic.csv")
    # 3. 역 라플라스 변환 기호 테이블 (주파수 s -> 시간 t)
    inv_laplace_data = [
        {"input": "1/s", "output": "1", "roc": "s > 0"},                  # 상수 단위계단 역변환
        {"input": "1/s**2", "output": "t", "roc": "s > 0"},               # 램프 함수 역변환
        {"input": "2/s**3", "output": "t**2", "roc": "s > 0"},            # 파라볼라 역변환
        {"input": "1/(s+1)", "output": "exp(-t)", "roc": "s > 0"},        # 지수 감쇠 역변환
        {"input": "1/(s+2)", "output": "exp(-2*t)", "roc": "s > 0"},      # 지수 감쇠 역변환
        {"input": "1/(s+1)**2", "output": "t*exp(-t)", "roc": "s > 0"},   # 시간-지수 혼합 역변환
        {"input": "1/(s**2 + 1)", "output": "sin(t)", "roc": "s > 0"},    # 사인 역변환
        {"input": "s/(s**2 + 1)", "output": "cos(t)", "roc": "s > 0"},    # 코사인 역변환
    ]
    df_inv_laplace = pd.DataFrame(inv_laplace_data)
    df_inv_laplace.to_csv(os.path.join(data_dir, "inverse_laplace_symbolic.csv"), index=False)
    print("생성 완료: inverse_laplace_symbolic.csv")
    # 4. 푸리에 코사인 변환 (Fourier Cosine Transform: t -> w)
    # 복소수 적분(scipy.quad의 한계)을 피해 순수 실수 영역에서 진화 가능한 푸리에 변환의 실수부 버전을 사용합니다.
    # 연산자: Integral(f_t * cos(w*t), (t, 0, oo))
    fourier_cosine_data = [
        {"input": "exp(-t)", "output": "1/(1 + w**2)"},               # 지수 감소
        {"input": "exp(-2*t)", "output": "2/(4 + w**2)"},             # 지수 감소 (a=2)
        {"input": "exp(-3*t)", "output": "3/(9 + w**2)"},             # 지수 감소 (a=3)
        {"input": "t*exp(-t)", "output": "(1 - w**2)/(1 + w**2)**2"}, # 시간-지수 혼합
        {"input": "t*exp(-2*t)", "output": "(4 - w**2)/(4 + w**2)**2"},
    ]
    df_fourier = pd.DataFrame(fourier_cosine_data)
    df_fourier.to_csv(os.path.join(data_dir, "fourier_cosine_symbolic.csv"), index=False)
    print("생성 완료: fourier_cosine_symbolic.csv")

    # 5. 푸리에 변환 (Hz 주파수 도메인 기준, t -> freq)
    # 연산자: Integral(f_t * cos(2 * pi * freq * t), (t, 0, oo))
    fourier_hz_data = [
        {"input": "exp(-t)", "output": "1 / (1 + (2*pi*freq)**2)"},
        {"input": "exp(-2*t)", "output": "2 / (4 + (2*pi*freq)**2)"},
        {"input": "exp(-3*t)", "output": "3 / (9 + (2*pi*freq)**2)"},
        {"input": "t*exp(-t)", "output": "(1 - (2*pi*freq)**2) / (1 + (2*pi*freq)**2)**2"}
    ]
    df_fourier_hz = pd.DataFrame(fourier_hz_data)
    df_fourier_hz.to_csv(os.path.join(data_dir, "fourier_hz_symbolic.csv"), index=False)
    print("생성 완료: fourier_hz_symbolic.csv")

    # 6. 복소 푸리에 변환 (Complex Fourier Transform: t -> f)
    # 연산자: Integral(f_t * exp(-I*w*t), (t, 0, oo)) 
    # (사용자가 언급한 exp(j2pift) 중 j는 Sympy에서 대문자 I로 표현됨. 통상적인 포워드 변환은 -j를 사용하므로 -I 적용)
    complex_fourier_data = [
        {"input": "exp(-t)", "output": "1/(1 + I*w)"},
        {"input": "exp(-2*t)", "output": "1/(2 + I*w)"},
        {"input": "exp(-3*t)", "output": "1/(3 + I*w)"},
        {"input": "t*exp(-t)", "output": "1/(1 + I*w)**2"},
        {"input": "t*exp(-2*t)", "output": "1/(2 + I*w)**2"}
    ]
    df_complex_fourier = pd.DataFrame(complex_fourier_data)
    df_complex_fourier.to_csv(os.path.join(data_dir, "complex_fourier_symbolic.csv"), index=False)
    print("생성 완료: complex_fourier_symbolic.csv")

    # 7. 위상 편이 역 푸리에 변환 (Phase-Shifted Inverse Fourier Transform)
    # 연산자: Integral(f_t * exp(I * (omega * t + a)), (t, 0, oo))
    phase_shifted_inv_fourier_data = [
        {"input": "exp(-t)", "output": "exp(I*a) / (1 - I*omega)"},
        {"input": "exp(-2*t)", "output": "exp(I*a) / (2 - I*omega)"},
        {"input": "t*exp(-t)", "output": "exp(I*a) / (1 - I*omega)**2"}
    ]
    df_phase_shifted = pd.DataFrame(phase_shifted_inv_fourier_data)
    df_phase_shifted.to_csv(os.path.join(data_dir, "phase_shifted_inv_fourier_symbolic.csv"), index=False)
    print("생성 완료: phase_shifted_inv_fourier_symbolic.csv")

if __name__ == "__main__":
    create_symbolic_datasets()
