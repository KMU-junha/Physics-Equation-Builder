# pyre-ignore-all-errors
"""
SINDy(Sparse Identification of Nonlinear Dynamics) 및 기호 회귀 엔진과 
베이지안 최적화(gp_minimize)를 사용하여 데이터로부터 수식 후보군을 자동으로 탐색하는 모듈입니다.
베이지안 회귀/GPR을 통한 불확실성(오차 분포) 추정 메타데이터도 함께 반환합니다.
"""

import re
import numpy as np
import pandas as pd
from sympy import Symbol, Eq
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
from sympy import lambdify

import pysindy as ps
from skopt import gp_minimize
from skopt.space import Real
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# ── 전역 변수 (베이지안 최적화 목적함수에서 데이터에 접근하기 위해 사용) ──────────
_X_data: np.ndarray = np.array([])
_t_data: np.ndarray = np.array([])
_v_data: np.ndarray = np.array([])
_v_surrogate: np.ndarray = np.array([])
_feature_names: list = []


def _get_sindy_library():
    """
    다항식, 푸리에(삼각함수), 사용자 정의(지수, 역수, 분수 등) 라이브러리를 결합하여 반환합니다.
    (물리적 차원 검증보다는 순수 Fitting 능력을 극대화하기 위해 라이브러리를 대폭 확장)
    """
    # 1. 다항식 라이브러리 (3차까지 확장)
    poly_lib = ps.PolynomialLibrary(degree=3, include_bias=True)
    
    # 2. 푸리에(삼각함수) 라이브러리 (진동 데이터 등의 비선형 주기성에 강점)
    fourier_lib = ps.FourierLibrary(n_frequencies=2)
    
    # 3. 사용자 정의 라이브러리 (단일/다중 변수의 분수 및 지수)
    custom_functions = [
        lambda x: np.exp(x),
        lambda x: np.exp(-x),
        lambda x: np.exp(-x) * np.sin(x),
        lambda x: np.exp(-x) * np.sin(2.0*x),
        lambda x: np.exp(-x) * np.sin(3.0*x),
        lambda x: np.exp(-x) * np.sin(4.0*x),
        lambda x: np.exp(-0.4*x) * np.sin(4.0*x),  # damped oscillator GT
        lambda x: np.exp(-0.5*(x - 3.0)**2) * np.sin(6.0*x),  # tone burst GT
        lambda x: np.exp(-0.5*(x - 3.0)**2) * np.cos(6.0*x),  # tone burst cosine variant
        lambda x: np.exp(-0.5*(x - 3.0)**2),                   # Gaussian envelope only
        lambda x: 1.0 / (np.abs(x) + 1e-6)
    ]
    custom_names = [
        lambda x: 'exp(' + x + ')',
        lambda x: 'exp(-' + x + ')',
        lambda x: 'exp(-' + x + ')*sin(' + x + ')',
        lambda x: 'exp(-' + x + ')*sin(2*' + x + ')',
        lambda x: 'exp(-' + x + ')*sin(3*' + x + ')',
        lambda x: 'exp(-' + x + ')*sin(4*' + x + ')',
        lambda x: 'exp(-0.4*' + x + ')*sin(4*' + x + ')',
        lambda x: 'exp(-0.5*(' + x + '-3)^2)*sin(6*' + x + ')',
        lambda x: 'exp(-0.5*(' + x + '-3)^2)*cos(6*' + x + ')',
        lambda x: 'exp(-0.5*(' + x + '-3)^2)',
        lambda x: '1 / ' + x
    ]
    
    custom_lib = ps.CustomLibrary(
        library_functions=custom_functions,
        function_names=custom_names
    )
    
    return ps.GeneralizedLibrary([poly_lib, fourier_lib, custom_lib])


def parse_sindy_equation(eq_str: str, local_dict: dict):
    """
    SINDy가 출력한 원시 문자열을 Sympy가 이해할 수 있는 수식으로 파싱하는 견고한 정규식 처리 함수입니다.
    예: "5.0159 1 + 2.0 d" -> "5.0159 + 2.0 * d"
    """
    if not eq_str or eq_str.strip() == "":
        return None
    
    # 1. 숫자 뒤에 공백 후 상수 1이 나오는 현상 제거 (예: "5.0159 1" -> "5.0159")
    parsed_str = re.sub(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)\s+1(?!\w)', r'\1', eq_str)
    
    # 2. 숫자와 문자열 기호 사이에 연산자가 빠진 경우 곱셈(*) 추가 (예: "5.0159 d" -> "5.0159 * d")
    parsed_str = re.sub(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)\s+([A-Za-z_]\w*)', r'\1*\2', parsed_str)
    parsed_str = re.sub(r'(\d+\.?\d*(?:[eE][+-]?\d+)?)([A-Za-z_]\w*)', r'\1*\2', parsed_str)
    
    # 3. 불필요한 공백 정리
    parsed_str = re.sub(r'\s+', ' ', parsed_str).strip()
    
    try:
        # 암묵적 곱셈 및 ^ 거듭제곱을 허용하는 파서 트랜스폼 적용
        transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
        expr = parse_expr(parsed_str, local_dict=local_dict, transformations=transformations)
        return expr
    except Exception as e:
        print(f"  [Parse Error] 수식 문자열 '{eq_str}' 파싱 실패. (정리된 문자열: '{parsed_str}', 사유: {e})")
        return None


def _sindy_mse_objective(params: list) -> float:
    """
    베이지안 최적화 목적함수: SINDy threshold를 조정하며 오차(MSE)를 최소화합니다.
    물리 차원 등은 무시하고 오직 곡선 적합(Fitting) 성능에 집중합니다.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    threshold = params[0]
    try:
        optimizer = ps.STLSQ(threshold=threshold, normalize_columns=True)
        lib = _get_sindy_library()
        model = ps.SINDy(feature_library=lib, optimizer=optimizer)
        
        # SINDy 피팅: X는 독립변수(input), x_dot은 타겟값(output) (Surrogate Data 사용)
        model.fit(_X_data, t=_t_data, x_dot=_v_surrogate, feature_names=_feature_names)
        pred = model.predict(_X_data)
        
        # 타겟 예측값
        v_pred = pred[:, 0]
        
        # 1. 평가지표: MSE (Surrogate Data 대비 오차)
        mse = float(np.mean((_v_surrogate.flatten() - v_pred.flatten()) ** 2))
        if not np.isfinite(mse):
            return 1e6
            
        # 2. 복잡도 페널티 (물리적 제약 무시, 단순히 항이 너무 많아지는 것만 방지)
        coef = model.optimizer.coef_[0]
        active_coefs = np.abs(coef[coef != 0])
        penalty = 0.0
        
        if len(active_coefs) == 0:
            return 1e9  # 과소적합 방지
            
        # Fitting에 집중하므로 항이 6개를 초과할 때만 가벼운 페널티 부과
        if len(active_coefs) > 6:
            penalty += (len(active_coefs) - 6) * 0.1
                
        total_score = mse + penalty
        return total_score if np.isfinite(total_score) else 1e6
        
    except Exception:
        return 1e6


def compute_probabilistic_distribution(t: np.ndarray, y: np.ndarray):
    """
    가우시안 프로세스 회귀(GPR)를 결합하여 각 시간(time-point) 축에 대한
    평균 예측값과 오차 범위(분산)를 확률 변수로 도출합니다.
    시각화 단계에서 사용될 수 있도록 메타데이터 형태로 반환합니다.
    """
    # 시간축 t 기반으로 추이를 파악하기 위해 GPR을 활용
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    # y값의 스케일 차이를 안정적으로 처리하기 위해 normalize_y=True 추가
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=3, normalize_y=True)
    
    t_reshaped = t.reshape(-1, 1)
    y_flat = y.flatten()  # 브로드캐스팅 오류(예: 150x150 배열 생성) 방지를 위해 1D로 평탄화
    
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpr.fit(t_reshaped, y_flat)
        
        y_mean, y_std = gpr.predict(t_reshaped, return_std=True)
        return {
            "time_axis": t.tolist(),
            "mean": y_mean.tolist(),
            "std": y_std.tolist(),
            "upper_bound": (y_mean + 1.96 * y_std).tolist(), # 95% 신뢰구간 상한
            "lower_bound": (y_mean - 1.96 * y_std).tolist()  # 95% 신뢰구간 하한
        }
    except Exception as e:
        print(f"  [Probabilistic Error] 분포 계산 오류: {e}")
        # 실패시 베이스라인(데이터 자체) 반환
        y_list = y_flat.tolist()
        return {
            "time_axis": t.tolist(),
            "mean": y_list,
            "std": np.zeros_like(y_flat).tolist(),
            "upper_bound": y_list,
            "lower_bound": y_list
        }


def generate_candidate_equations(df: pd.DataFrame, engine: str = "sindy") -> list:
    """
    SINDy(또는 확장된 타 엔진) + 베이지안 최적화를 사용하여 데이터로부터 물리 수식 후보군을 탐색합니다.

    Args:
        df (pd.DataFrame): 'd', 't', 'v' 열을 포함하는 운동학 데이터프레임.
        engine (str): 탐색에 사용할 기호 회귀 엔진 ('sindy' 또는 'pysr' 등등 동적 확장 가능)

    Returns:
        list: (sympy.Eq, float, dict) 튜플의 리스트.
              (수식 객체, 해당 수식의 MSE, 오차 분포 메타데이터)
    """
    global _X_data, _t_data, _v_data, _feature_names

    # ── 1. 데이터 준비 ──────────────────────────────────────────────────────────
    # 새로운 데이터 포맷 (input, output) - 대수 방정식(Algebraic Equation) 탐색용
    if "input" in df.columns and "output" in df.columns:
        x_arr = df["input"].to_numpy(dtype=float)
        y_arr = df["output"].to_numpy(dtype=float)
        
        # ROC는 Region of Convergence(수렴 영역)이므로 피팅 데이터로 쓰지 않음
        
        # SINDy를 대수식(output = f(input)) 탐색용으로 우회: 
        # 특징 벡터(X)는 input만, 타겟(x_dot 자리)에 output 배치
        _X_data = x_arr.reshape(-1, 1)  # [input]
        _t_data = x_arr  # x_axis 역할 (가우시안 프로세스 플로팅용)
        _v_data = y_arr.reshape(-1, 1)  # 타겟 (output)
        _feature_names = ["input"]
        
        # 평가/파싱용 변수 할당
        target_sym = Symbol("output")
        indep_sym = Symbol("input")
        target_dot_sym = target_sym  # 이제 출력의 타겟은 수렴영역(ROC)이나 변화율이 아닌 output 자체
        parse_dict = {"input": indep_sym}
        
    else:
        # 레거시 포맷 (d, t, v) 호환 유지 (기존 동역학 피팅용)
        d_arr = df["d"].to_numpy(dtype=float)
        t_arr = df["t"].to_numpy(dtype=float)
        v_arr = df["v"].to_numpy(dtype=float)

        _X_data = np.column_stack([d_arr, t_arr])
        _t_data = t_arr
        _v_data = v_arr.reshape(-1, 1)
        _feature_names = ["d", "t"]
        
        # 평가/파싱용 변수 할당
        target_sym = Symbol("d")
        indep_sym = Symbol("t")
        target_dot_sym = Symbol("v")
        parse_dict = {"d": target_sym, "t": indep_sym}
        
        # 하위 코드 검증(MSE) 로직용 배열 매핑
        x_arr = t_arr
        y_arr = d_arr

    candidate_pool = []
    
    # ── [엔진 1] SINDy 활용 파이프라인 ───────────────────────────────────────
    if engine.lower() in ["sindy", "all"]:
        # 오차 분포 도출 및 Surrogate 생성 (가우시안 프로세스 활용)
        print("  [Actor] 가우시안 프로세스 회귀(GPR)로 Surrogate 데이터 생성 중...")
        prob_dist = compute_probabilistic_distribution(_t_data, _v_data)
        
        # SINDy에 원본 노이즈 데이터 대신 GPR 평균값(Surrogate)을 주입
        global _v_surrogate
        _v_surrogate = np.array(prob_dist["mean"]).reshape(-1, 1)

        print("  [Actor] 베이지안 최적화로 SINDy 하이퍼파라미터 탐색 중...")
        search_space = [Real(1e-3, 2.0, name="threshold")]

        # 베이지안 최적화: threshold 탐색
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = gp_minimize(
                func=_sindy_mse_objective,
                dimensions=search_space,
                n_calls=15,
                n_initial_points=5,
                random_state=42,
                verbose=False,
            )

        best_threshold = result.x[0]
        best_cost = result.fun
        print(f"  [Actor] SINDy 최적 threshold={best_threshold:.4f}, 최소 Cost(MSE+Penalty)={best_cost:.6f}")

        # 최적 파라미터로 다시 피팅
        optimizer = ps.STLSQ(threshold=best_threshold, normalize_columns=True)
        lib = _get_sindy_library()
        model = ps.SINDy(feature_library=lib, optimizer=optimizer)

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(_X_data, t=_t_data, x_dot=_v_surrogate, feature_names=_feature_names)
            
            sindy_equations = model.equations(precision=4)
            print(f"  [Actor] SINDy 발견 수식 원본 (문자열): {sindy_equations}")
        except Exception as e:
            print(f"  [Actor] SINDy 피팅 실패: {e}")
            sindy_equations = []

        # Sympy 파싱 준비
        for i, eq_str in enumerate(sindy_equations):
            if not eq_str:
                continue

            # 0번째 상태변수(예: output 또는 d)의 도함수에 대한 방정식만 처리
            if i == 0:
                rhs_expr = parse_sindy_equation(eq_str, parse_dict)
                
                if rhs_expr is not None:
                    eq_obj = Eq(target_dot_sym, rhs_expr)
                    
                    # MSE 검증
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if "input" in df.columns:
                                # 대수식은 독립변수(input)만 필요
                                pred_func = lambdify((indep_sym,), rhs_expr, modules="numpy")
                                v_pred = pred_func(x_arr)
                            else:
                                # 레거시 동역학 포맷
                                pred_func = lambdify((target_sym, indep_sym), rhs_expr, modules="numpy")
                                v_pred = pred_func(y_arr, x_arr)
                                
                            # v_pred가 스칼라 상수(예: 1.5)로 나올 경우 배열로 변환
                            if np.isscalar(v_pred):
                                v_pred = np.full_like(x_arr, float(v_pred))
                                
                        mse_val = _compute_mse(_v_surrogate.flatten(), v_pred.flatten())
                        
                        candidate_pool.append((eq_obj, mse_val, prob_dist))
                    except Exception as e:
                        print(f"  [Actor] 수식 검증 실패 ({eq_str}): {e}")

    # ── [엔진 2] gplearn 기반 기호 회귀 엔진 분기 ─────────────────────────────────
    elif engine.lower() == "gplearn":
        print("  [Actor] gplearn (유전 프로그래밍) 엔진 분기 진입...")
        
        # 기본 함수 외에도 필요한 연산들을 추가할 수 있습니다.
        function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv']
        
        # SymbolicRegressor 초기화
        est_gp = SymbolicRegressor(population_size=1000,
                                   generations=20, stopping_criteria=0.01,
                                   p_crossover=0.7, p_subtree_mutation=0.1,
                                   p_hoist_mutation=0.05, p_point_mutation=0.1,
                                   max_samples=0.9, verbose=0,
                                   parsimony_coefficient=0.01, random_state=42,
                                   function_set=function_set)
        
        try:
            print("  [Actor] gplearn 진화 탐색 시작...")
            est_gp.fit(_X_data, _v_data)
            
            # gplearn이 찾은 최고의 식 문자열
            best_expr_str = str(est_gp._program)
            print(f"  [Actor] gplearn 발견 수식 원본 (문자열): {best_expr_str}")
            
            # 오차 분포 도출
            prob_dist = compute_probabilistic_distribution(_t_data, _v_data)
            
            # Sympy 파싱 준비
            # gplearn은 입력 피처를 X0, X1, ... 로 표현합니다.
            # 우리의 _X_data 에는 [y_arr, x_arr] 순서로 들어갔으므로 X0=target, X1=indep 입니다.
            local_dict = {"X0": target_sym, "X1": indep_sym}
            
            from sympy import sympify
            
            # gplearn 특유의 연산자를 SymPy 함수나 연산으로 직접 매핑
            import sympy
            
            def inv_func(x):
                return 1.0 / x
                
            def neg_func(x):
                return -x
                
            def abs_func(x):
                return sympy.Abs(x)
                
            def div_func(x, y):
                return x / y
                
            def mul_func(x, y):
                return x * y
                
            def add_func(x, y):
                return x + y
                
            def sub_func(x, y):
                return x - y
                
            def sqrt_func(x):
                return sympy.sqrt(x)
                
            def log_func(x):
                return sympy.log(x)

            local_dict = {
                "X0": target_sym, 
                "X1": indep_sym,
                "inv": inv_func,
                "neg": neg_func,
                "abs": abs_func,
                "div": div_func,
                "mul": mul_func,
                "add": add_func,
                "sub": sub_func,
                "sqrt": sqrt_func,
                "log": log_func
            }
            
            # 파서로 SymPy 객체 변환 시도
            rhs_expr = sympify(best_expr_str, locals=local_dict)
            
            if rhs_expr is not None:
                eq_obj = Eq(target_dot_sym, rhs_expr)
                
                # 예측 및 MSE 검증
                if "input" in df.columns:
                    pred_func = lambdify((indep_sym,), rhs_expr, modules="numpy")
                    v_pred = pred_func(x_arr)
                else:
                    pred_func = lambdify((target_sym, indep_sym), rhs_expr, modules="numpy")
                    v_pred = pred_func(y_arr, x_arr)
                    
                if np.isscalar(v_pred):
                    v_pred = np.full_like(x_arr, float(v_pred))
                    
                mse_val = _compute_mse(_v_data.flatten(), v_pred.flatten())
                
                candidate_pool.append((eq_obj, mse_val, prob_dist))
                
        except Exception as e:
            print(f"  [Actor] gplearn 심볼릭 회귀 실패: {e}")

    # ── [엔진 3] PySR 등 외부 엔진 확장성을 위한 구조 ───────────────────────
    elif engine.lower() == "pysr":
        print("  [Actor] PySR 엔진 분기 진입 (향후 구현용)")
        pass

    # ── Fall-back (실패 시 기본 후보군 셋업) ──────────────────────────────
    if not candidate_pool:
        print("  [Actor] SINDy가 유효한 식을 찾지 못해 기본 후보군을 분산 메타데이터와 함께 반환합니다.")
        
        prob_dist = compute_probabilistic_distribution(x_arr, _v_data.flatten())
        
        if "input" in df.columns:
            base_eqs = [
                (Eq(target_dot_sym, indep_sym**2), _compute_mse(_v_data.flatten(), x_arr**2)),
                (Eq(target_dot_sym, indep_sym*2), _compute_mse(_v_data.flatten(), x_arr*2)),
                (Eq(target_dot_sym, indep_sym), _compute_mse(_v_data.flatten(), x_arr))
            ]
        else:
            base_eqs = [
                (Eq(target_dot_sym, target_sym / indep_sym), _compute_mse(_v_data.flatten(), y_arr / x_arr)),
                (Eq(target_dot_sym, target_sym * indep_sym), _compute_mse(_v_data.flatten(), y_arr * x_arr)),
                (Eq(target_dot_sym, target_sym + indep_sym), _compute_mse(_v_data.flatten(), y_arr + x_arr))
            ]
        
        for eq_obj, mse_val in base_eqs:
            candidate_pool.append((eq_obj, mse_val, prob_dist))

    print(f"  [Actor] 최종 {len(candidate_pool)}개 튜플 후보 반환 완료")
    return candidate_pool


def _compute_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    두 numpy 배열 간의 평균 제곱 오차(MSE)를 계산합니다.
    (0으로 나누는 등의 문제로 발생한 NaN 값들은 제외하고 계산합니다)
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff = actual - predicted
        diff = diff[~np.isnan(diff)]
        if len(diff) == 0:
            return float('inf')
        return float(np.mean(diff ** 2))
