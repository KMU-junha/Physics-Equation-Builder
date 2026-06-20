# pyre-ignore-all-errors
"""
전체 파이프라인을 실행하는 메인 실행 파일입니다.
CSV 데이터를 로드하고 SINDy+베이지안 최적화 Actor → Critic → Pareto 순서로 실행합니다.
"""

import os
import sys
import time
import math

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


# 프로젝트 루트(physics_equation_builder)를 Python 경로에 추가하여 src 모듈을 찾을 수 있게 합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # pyre-ignore[21]
from sympy import Eq, Symbol
from src.dataset_analyzer import detect_dataset_type
from src.actor_generator import generate_candidate_equations
from src.meta_symbolic_engine import generate_symbolic_operator
from src.critic_validator import check_dimensional_homogeneity
from src.pareto_selector import select_best_equations


def calculate_fitness_score(mse: float, complexity: int) -> float:
    """MSE와 복잡도를 100점 만점의 적합도(Fitness)로 변환"""
    if mse < 1e-7:
        base_score = 100.0
    else:
        # MSE 0.002 -> ~98점, MSE 0.1 -> ~36점
        base_score = 100.0 * math.exp(-10.0 * mse)
        
    # 복잡도가 8을 초과할 때부터 약간의 패널티 부과
    penalty = max(0, complexity - 8) * 0.5
    return max(0.0, min(100.0, base_score - penalty))


def run_pipeline(data_path: str):
    start_time = time.time()
    print(f"\n{Colors.CYAN}{'='*70}")
    print(f"{Colors.BOLD}[AI 물리 수식 탐색 파이프라인 가동] 분석 대상: {os.path.basename(data_path)}")
    print(f"{'='*70}{Colors.RESET}")

    # 1. 데이터 로드 및 분석
    print(f"\n{Colors.BLUE}[1단계] 데이터 로드 중...{Colors.RESET}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"{Colors.RED}  데이터 로드 실패: {e}{Colors.RESET}")
        return
        
    data_type = detect_dataset_type(df)
    print(f"  로드 완료: {len(df)}행, 열: {list(df.columns)}")
    print(f"  감지된 데이터 형태: {Colors.YELLOW}{data_type}{Colors.RESET}")

    # 2. Actor (수식 후보 탐색)
    print(f"\n{Colors.BLUE}[2단계] Actor 구조 탐색 중...{Colors.RESET}")
    
    if data_type == "SYMBOLIC":
        print(f"  {Colors.MAGENTA}[Router] 심볼릭 데이터가 감지되었습니다. Meta-Symbolic 연산자 추론 엔진을 가동합니다. (패널티 가중치: {_PENALTY_WEIGHT}){Colors.RESET}")
        best_transform = generate_symbolic_operator(df, pop_size=_POP_SIZE, generations=_GENERATIONS, penalty_multiplier=_PENALTY_WEIGHT, hints=_HINTS)
        print(f"\n  {Colors.GREEN}{Colors.BOLD}[최종 추론 물리 방정식 (Physical Operator)]{Colors.RESET}")
        print(f"         {Colors.YELLOW}{Colors.BOLD}최종수식: {best_transform}{Colors.RESET}")
        elapsed_time = time.time() - start_time
        print(f"\n{Colors.CYAN}" + "=" * 70)
        print(f"[메타 기호 연산자 탐색 종료] 총 소요 시간: {elapsed_time:.2f}초")
        print("=" * 70 + f"{Colors.RESET}\n")
        return # 심볼릭은 Critic/Pareto 등 현재 불필요
    else:
        print("  [Router] 연속 수치 데이터가 감지되었습니다. SINDy 역학 피팅 엔진을 가동합니다.")
        # SINDy는 기본적으로 미분과 빠른 대수 피팅 특화
        try:
            candidate_results = generate_candidate_equations(df, engine='sindy')
        except Exception as e:
            print(f"  Actor 실행 중 에러: {e}")
            return
        print(f"  총 {len(candidate_results)}개의 수식 후보가 생성되었습니다.")

    # ── 3. Critic: 차원(물리법칙) 검증 ───────────────────────────────────────────
    print("\n[3단계] Critic (차원 분석) 검증 중...")
    valid_candidates = []
    
    from sympy.physics.units import meter, second
    
    v_sym = Symbol('v')
    d_sym = Symbol('d')
    t_sym = Symbol('t')
    target_syms = ['v', 'a']
    
    # 단위 매핑 딕셔너리
    subs = {
        v_sym: meter / second,
        d_sym: meter,
        t_sym: second,
    }

    print("-" * 60)
    for eq, mse, meta in candidate_results:
        # 방정식의 좌/우변에 단위를 대입 (Subs)
        lhs_with_units = eq.lhs.subs(subs)
        rhs_with_units = eq.rhs.subs(subs)
        
        is_valid = check_dimensional_homogeneity(lhs_with_units, rhs_with_units)
        
        eq_str_print = f"{eq.lhs} = {eq.rhs}" if isinstance(eq, Eq) else str(eq)
        status_str = "✅ 통과" if is_valid else "❌ 실패"
        print(f"  방정식 | {eq_str_print:^30} | MSE={mse:.4f} | 차원: {status_str}")
        
        if is_valid:
            valid_candidates.append((eq, mse, meta))
    print("-" * 60)
    print(f"  통과: {len(valid_candidates)}개 / 전체: {len(candidate_results)}개")

    # ── 4. Pareto 선택기 ────────────────────────────────────────────────────────
    if not valid_candidates:
        print("\n[4단계] 차원 검증을 통과한 수식이 없어 Pareto 평가를 건너뜁니다.")
        return

    print("\n[4단계] Pareto 선택기 (복잡도 vs MSE) 평가 중...")
    passed_eqs = [tup[0] for tup in valid_candidates]
    passed_mses = [tup[1] for tup in valid_candidates]
    
    best_eqs = select_best_equations(passed_eqs, passed_mses)
    
    # 최종 결과 출력
    print("\n  [Pareto 평가 결과]")
    print(f"    {'수식':^30} {'복잡도':^10} {'MSE':^10} {'점수':^10}")
    print("  " + "-" * 65)
    for i, (eq, (complexity, err)) in enumerate(best_eqs.items()):
        if isinstance(eq, Eq):
            eq_str = f"{eq.lhs} = {eq.rhs}"
        else:
            eq_str = str(eq)
            
        score = calculate_fitness_score(err, complexity)
        print(f"    {eq_str:^30} {complexity:^10} {err:^10.6f} {score:^10.2f}")
        
        # 첫 번째 항목(최적해)을 api.py가 캡처할 수 있도록 명시적 출력
        if i == 0:
            print(f"         최종수식: {eq_str}")
            print(f"         적합도: {score:.4f}")
            if err < 1e-5:
                print("         완벽한 해 발견!")
                
            best_meta = None
            for veq, vmse, vmeta in valid_candidates:
                if veq == eq:
                    best_meta = vmeta
                    break
                    
            if best_meta:
                import json
                from sympy import lambdify
                import numpy as np
                
                t_axis = np.array(best_meta["time_axis"])
                if "output" in df.columns:
                    y_actual = df["output"].values.tolist()
                else:
                    y_actual = df["v"].values.tolist()
                    
                try:
                    if "input" in df.columns:
                        pred_func = lambdify(Symbol("input"), eq.rhs, modules="numpy")
                        y_pred = pred_func(t_axis)
                    else:
                        pred_func = lambdify((Symbol("d"), Symbol("t")), eq.rhs, modules="numpy")
                        y_pred = pred_func(df["d"].values, t_axis)
                    if np.isscalar(y_pred):
                        y_pred = np.full_like(t_axis, float(y_pred))
                    y_pred_list = y_pred.tolist()
                except Exception:
                    y_pred_list = []
                    
                plot_data = {
                    "time_axis": best_meta["time_axis"],
                    "y_actual": y_actual,
                    "y_pred": y_pred_list,
                    "gpr_mean": best_meta.get("mean", []),
                    "gpr_std": best_meta.get("std", [])
                }
                print(f"[PLOT_DATA] {json.dumps(plot_data)}")

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🏁 파이프라인 탐색을 성공적으로 마쳤습니다! ⏱️ 총 소요 시간: {elapsed_time:.2f}초")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Physics Equation Builder')
    parser.add_argument('csv', nargs='?',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'fraunhofer_k_symbolic.csv'),
                        help='CSV data path')
    parser.add_argument('--pop-size', type=int, default=1000)
    parser.add_argument('--generations', type=int, default=15)
    parser.add_argument('--penalty-weight', type=float, default=1.0)
    parser.add_argument('--hints', type=str, default="{}")
    args = parser.parse_args()
    _POP_SIZE = args.pop_size
    _GENERATIONS = args.generations
    _PENALTY_WEIGHT = args.penalty_weight
    import json
    try:
        _HINTS = json.loads(args.hints)
    except:
        _HINTS = {}
    run_pipeline(args.csv)

