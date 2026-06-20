import random
import sympy
from sympy import Symbol, Add, Mul, Pow, exp, sin, cos, Integral, Derivative, oo, simplify, lambdify
from typing import List, Dict, Callable, Any
import numpy as np
import os
import concurrent.futures
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# ==============================================================================
# 듀얼 트랙 시스템 - 연산자 추론 트랙 (Meta Symbolic Engine)
# ==============================================================================
class GlassBoxLogger:
    def __init__(self, filepath, dataset_type="SYMBOLIC"):
        self.filepath = filepath
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("# Meta-Symbolic Evolution Trace (Log)\n\n")
            f.write(f"### 데이터 진단\n- 감지된 데이터 형태: `{dataset_type}`\n\n---\n")
            
    def log_deduction(self, templates):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("## 1. 공리 연역 (Axiomatic Deduction) 판정 결과\n")
            for t in templates:
                f.write(f"`{str(t)}`\n")
            f.write("\n---\n")
            
    def log_terminal_pool(self, terminal_pool):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"## 2. 동적 유전자 풀 (Dynamic Gene Pool)\n- 변수 및 상수: `{terminal_pool}`\n\n---\n")
            
    def log_generation(self, depth, gen, best_score, best_operator, elapsed_time=0.0):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"### [전역 최고해 갱신 (Global Best Update)]\n")
            f.write(f"- 진행 스텝: Depth {depth} | Gen {gen}\n")
            f.write(f"- 적합도 점수(Fitness Score): {best_score:.4f}\n")
            f.write(f"`{str(best_operator)}`\n\n")

    def log_dimension_chain(self, required_var_count, computed_min_depth, var_names, pop_size, generations_per_depth):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"## 3. 탐색 하이퍼파라미터 (Search Hyperparameters)\n")
            f.write(f"- 필수 도메인 변수: `{var_names}` ({required_var_count}개)\n")
            f.write(f"- 최소 강제 탐색 깊이: `{computed_min_depth}`\n")
            f.write(f"- 세대별 개체 수(Population): `{pop_size}`\n")
            f.write(f"- 층위별 최대 세대 수: `{generations_per_depth}`회\n\n---\n")

    def log_successive_deepening_start(self, depth, seed_count, current_best_score):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"## [Depth {depth}] 점진적 심층 탐색 시작\n")
            if seed_count > 0:
                f.write(f"- 빔 서치(Beam Search) 확장: 이전 깊이 우수 커널 `{seed_count}`개 진화 씨앗 활용\n")
            f.write("\n")

    def log_generation_summary(self, depth, gen, total_evaluated, cache_hit_rate, generation_best_score):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"- [세대 {gen} 완료] 누적 탐색: `{total_evaluated}`개 | 세대 최고점: `{generation_best_score:.4f}` | 중복 캐시 히트율: `{cache_hit_rate:.1f}`%\n")

    def log_completion(self, elapsed_time, final_operator, final_score, node_count):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(f"\n---\n## 4. 최종 탐색 요약 (Final Output)\n")
            f.write(f"- 총 탐색 소요 시간: `{elapsed_time:.2f}`초\n")
            if final_operator:
                f.write(f"- 수식 복잡도 (AST Node Size): `{node_count}`\n")
                f.write(f"\n### [추론된 물리 방정식 (Physical Operator)] Score: {final_score:.4f}\n")
                f.write(f"`{str(final_operator)}`\n\n")

    def log_shortcut_result(self, operator, dataset, elapsed_time):
        """쇼트컷 성공 시 탐색 경로·검증 테이블·연산자 구조를 ## 5 섹션으로 기록."""
        f_t_sym = sympy.Symbol('f_t')
        is_derivative_op = operator.has(sympy.Derivative)
        with open(self.filepath, 'a', encoding='utf-8') as f:

            f.write("\n---\n## 5. 결과 해석 및 탐색 경로 분석\n\n")

            # ── 5.1 탐색 경로 ───────────────────────────────────────────────────────
            f.write("### 5.1 탐색 경로 (Discovery Path)\n")
            if is_derivative_op:
                f.write("- 방법: 미분 연산자 심볼릭 쇼트컷 — 기호 미분 검증으로 연산자 직접 확정 (GA 탐색 생략)\n")
                f.write("- 원리: 각 입출력 쌍에서 `diff(input, var, n)` 수행 후 output과 일치 여부 확인,\n")
                f.write("  모든 행에서 일관성이 확인되면 해당 미분 연산자를 확정\n")
            else:
                f.write("- 방법: 유리식 심볼릭 쇼트컷 — 직접 치환으로 연산자 추출 (GA 탐색 생략)\n")
                f.write("- 원리: 각 입출력 쌍에서 `output.subs(f_t → □)` 수행 후 동일한 □의 함수 추출,\n")
                f.write("  모든 행에서 일관성이 확인되면 해당 함수를 연산자로 확정\n")
            f.write(f"- 소요 시간: `{elapsed_time:.3f}초`\n\n")

            # ── 5.2 입출력 검증 테이블 ─────────────────────────────────────────────
            f.write("### 5.2 입출력 일관성 검증 (Row-wise Verification)\n")
            f.write("|  # | 입력 `f_t` | 정답 | 연산자 적용 결과 | 일치 |\n")
            f.write("|:--:|---|---|---|:--:|\n")
            all_match = True
            for i, row in enumerate(dataset):
                in_e  = row.get('input_expr')
                out_e = row.get('target_expr')
                try:
                    substituted = operator.subs(f_t_sym, in_e)
                    # 미분형은 .doit()으로 평가 후 비교 (미평가 Derivative는 simplify가 0 반환 실패)
                    if is_derivative_op:
                        applied = substituted.doit()
                    else:
                        applied = sympy.cancel(substituted)
                    diff    = sympy.simplify(applied - out_e)
                    match   = "✅" if diff == 0 else "⚠️"
                    if diff != 0:
                        all_match = False
                except Exception:
                    applied = "계산 불가"
                    match   = "❓"
                    all_match = False
                f.write(f"| {i+1} | `{in_e}` | `{out_e}` | `{applied}` | {match} |\n")
            f.write("\n")
            verdict = "✅ 모든 행 완전 일치 — 연산자 확정" if all_match else "⚠️ 일부 행 불일치 — 근사 추출"
            f.write(f"> {verdict}\n\n")

            # ── 5.3 연산자 구조 분석 ───────────────────────────────────────────────
            f.write("### 5.3 연산자 구조 분석 (Structural Analysis)\n")
            f.write(f"- 전개형: `{operator}`\n")
            if is_derivative_op:
                for d_atom in operator.atoms(sympy.Derivative):
                    diff_var = d_atom.variables
                    order = len(diff_var)
                    f.write(f"- 구조: {order}계 미분 연산자 (Derivative Operator)\n")
                    f.write(f"  - 미분 변수: `{diff_var[0]}`\n")
                    f.write(f"  - 미분 차수: {order}계\n")
                    break
            else:
                try:
                    factored = sympy.factor(operator)
                    if str(factored) != str(operator):
                        f.write(f"- 인수분해형: `{factored}`\n")
                except Exception:
                    pass
                try:
                    numer, denom = sympy.fraction(sympy.cancel(operator))
                    if denom != 1:
                        f.write("- 구조: 유리식 (Rational Function)\n")
                        f.write(f"  - 분자: `{numer}`\n")
                        f.write(f"  - 분모: `{denom}`\n")
                    else:
                        try:
                            poly = sympy.Poly(operator, f_t_sym)
                            deg  = poly.degree()
                            f.write(f"- 구조: `f_t`에 대한 {deg}차 다항식 (Polynomial, degree={deg})\n")
                            terms = []
                            for d in range(deg, -1, -1):
                                c = poly.nth(d)
                                if c != 0:
                                    terms.append(f"  - `f_t^{d}`: 계수 = `{c}`")
                            f.write("\n".join(terms) + "\n")
                        except Exception:
                            f.write("- 구조: 대수식 (Algebraic Expression)\n")
                except Exception:
                    pass
            f.write("\n")

    def log_ga_result(self, operator, elapsed_time, total_evaluated, final_score):
        """GA 탐색 종료 후 탐색 경로 · 증명 결과 · 연산자 구조를 ## 5. 섹션으로 기록."""
        f_t_sym = sympy.Symbol('f_t')
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write("\n---\n## 5. 결과 해석 및 탐색 경로 분석\n\n")

            # ── 5.1 탐색 경로
            f.write("### 5.1 탐색 경로 (Discovery Path)\n")
            f.write("- 방법: 점진적 심층 탐색 + 빔 서치 GA (Successive Deepening + Beam Search)\n")
            f.write("- 원리: 최소 깊이부터 시작하여 빔 서치로 우수 커널을 씨앗으로 점진 확장,\n")
            f.write("  상위 엘리트에 대해 기호 증명(Symbolic Proof)으로 완전 일치 여부 검증\n")
            f.write(f"- 소요 시간: `{elapsed_time:.3f}초`\n")
            f.write(f"- 수식 평가 수: `{total_evaluated}개`\n")
            if final_score >= 99.9:
                f.write("- 기호 증명: ✅ 성공 — 닫힌 형태(Closed Form) 완전 일치 확인\n\n")
            else:
                f.write(f"- 기호 증명: ⚠️ 미완 — 수치 적합도 기반 최고해 (`{final_score:.4f}점`)\n\n")

            # ── 5.2 연산자 구조 분석
            f.write("### 5.2 연산자 구조 분석 (Structural Analysis)\n")
            if operator is None:
                f.write("- 탐색 실패 (코드 반환 없음)\n\n")
                return
            f.write(f"- 추론된 연산자: `{operator}`\n")
            try:
                if operator.has(sympy.Integral):
                    integrand = operator.args[0]
                    limits    = operator.args[1]
                    int_var   = limits[0]
                    lower     = limits[1]
                    upper     = limits[2]
                    f.write("- 구조: 적분형 연산자 (Integral Operator)\n")
                    f.write(f"  - 적분 변수: `{int_var}`\n")
                    f.write(f"  - 적분 구간: `[{lower}, {upper})`\n")
                    if integrand.has(f_t_sym):
                        try:
                            kernel = sympy.cancel(integrand / f_t_sym)
                            f.write(f"  - 피적분함수: `f_t * {kernel}`\n")
                            f.write(f"  - 커널 (Kernel): `{kernel}`\n")
                        except Exception:
                            f.write(f"  - 피적분함수: `{integrand}`\n")
                else:
                    try:
                        numer, denom = sympy.fraction(sympy.cancel(operator))
                        if denom != 1:
                            f.write("- 구조: 유리식 (Rational Function)\n")
                            f.write(f"  - 분자: `{numer}`\n")
                            f.write(f"  - 분모: `{denom}`\n")
                        else:
                            f.write("- 구조: 대수식 (Algebraic Expression)\n")
                    except Exception:
                        pass
            except Exception:
                pass
            f.write("\n")

c_light = Symbol('c', positive=True)
g_grav = Symbol('g', positive=True)
G_grav = Symbol('G', positive=True)
k_B = Symbol('k_B', positive=True)
h_bar = Symbol('hbar', positive=True)
Z_0 = Symbol('Z_0', positive=True)
mu_0 = Symbol('mu_0', positive=True)
eps_0 = Symbol('eps_0', positive=True)

s = Symbol('s')
t = Symbol('t', real=True)
freq = Symbol('f', positive=True)
omega = Symbol('omega', real=True)
theta = Symbol('theta', real=True)
period = Symbol('T', positive=True)
k_wave = Symbol('k', positive=True)
zeta = Symbol('zeta', positive=True)
tau = Symbol('tau', positive=True)

a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
f_t = Symbol('f_t')

# --- TUPLE AST FUNCTIONS ---

# Tan은 구조적으로 pi/2 주기 특이점 → 무한 적분 발산 보장 → 영구 제외
TRIG_OPS = frozenset({'Sin', 'Cos'})
DECAY_OPS = frozenset({'Exp', 'Pow'})

def random_tuple_operator(depth: int = 0, terminal_pool: list = None, min_depth: int = 0):
    # min_depth 미만이면 반드시 분기 연산자를 선택하도록 terminal weight을 0으로 눌러버림
    if depth < min_depth:
        terminal_weight = 0
    else:
        terminal_weight = max(5, 5 + (depth ** 3))
    # 0=Terminal 1=Add 2=Mul 3=Pow 4=Exp 5=Sin 6=Cos 7=Log
    node_type = random.choices([0, 1, 2, 3, 4, 5, 6, 7],
                               weights=[terminal_weight, 2, 4, 1, 1, 1, 1, 0.5])[0]

    if node_type == 0:
        return random.choice(terminal_pool)
    elif node_type == 1:
        return ('Add', random_tuple_operator(depth + 1, terminal_pool, min_depth), random_tuple_operator(depth + 1, terminal_pool, min_depth))
    elif node_type == 2:
        return ('Mul', random_tuple_operator(depth + 1, terminal_pool, min_depth), random_tuple_operator(depth + 1, terminal_pool, min_depth))
    elif node_type == 3:
        return ('Pow', random_tuple_operator(depth + 1, terminal_pool, min_depth), random.choice([-1, 1, 2]))
    elif node_type == 4:
        return ('Exp', random_tuple_operator(depth + 1, terminal_pool, min_depth))
    elif node_type == 5:
        return ('Sin', random_tuple_operator(depth + 1, terminal_pool, min_depth))
    elif node_type == 6:
        return ('Cos', random_tuple_operator(depth + 1, terminal_pool, min_depth))
    elif node_type == 7:
        return ('Log', random_tuple_operator(depth + 1, terminal_pool, min_depth))

def get_all_nodes_with_paths(tup, current_path=()):
    nodes = [(current_path, tup[0] if isinstance(tup, tuple) else type(tup), tup)]
    if isinstance(tup, tuple):
        for i, child in enumerate(tup[1:], 1):
            nodes.extend(get_all_nodes_with_paths(child, current_path + (i,)))
    return nodes

def replace_at_path(tup, path, new_node):
    if not path:
        return new_node
    if not isinstance(tup, tuple):
        return tup
    idx = path[0]
    replaced_child = replace_at_path(tup[idx], path[1:], new_node)
    new_tup = list(tup)
    new_tup[idx] = replaced_child
    return tuple(new_tup)

def tuple_homologous_crossover(tup1, tup2):
    nodes1 = get_all_nodes_with_paths(tup1)
    nodes2 = get_all_nodes_with_paths(tup2)
    if not nodes1 or not nodes2:
        return tup1
    random.shuffle(nodes1)
    for path1, type1, node1 in nodes1:
        matching = [n for n in nodes2 if n[1] == type1]
        if matching:
            if not path1 and len(matching) == 1 and not matching[0][0]:
                continue
            path2, type2, node2 = random.choice(matching)
            return replace_at_path(tup1, path1, node2)
    path1, _, _ = random.choice(nodes1)
    _, _, node2 = random.choice(nodes2)
    if path1:
        return replace_at_path(tup1, path1, node2)
    return tup1

def get_species_id(kernel_tuple, available_vars):
    # 고정 좌표 상수 (초월수 활용으로 우연한 해시 충돌 방지)
    sentinel_point = {
        's': 3.14159, 't': 2.71828, 'a': 1.41421, 'b': 1.61803, 
        'w': 0.57721, 'omega': 0.57721, 'x': 2.23606, 'y': 2.64575
    }
    base_dict = {v: sentinel_point.get(v, 1.11111 + 0.1*i) for i, v in enumerate(available_vars)}
    
    import numpy as np
    
    try:
        base_val = evaluate_tuple_ast(kernel_tuple, base_dict, math_module=np)
        if np.isnan(base_val) or np.isinf(base_val) or np.iscomplexobj(base_val):
            return "INVALID"
    except:
        return "INVALID"
        
    gradient_tuple = []
    h = 1e-4
    for var in available_vars:
        forward_dict = base_dict.copy()
        forward_dict[var] += h
        try:
            f_forward = evaluate_tuple_ast(kernel_tuple, forward_dict, math_module=np)
        except:
            f_forward = base_val
            
        backward_dict = base_dict.copy()
        backward_dict[var] -= h
        try:
            f_backward = evaluate_tuple_ast(kernel_tuple, backward_dict, math_module=np)
        except:
            f_backward = base_val
            
        derivative = (f_forward - f_backward) / (2 * h)
        gradient_tuple.append(round(derivative, 3))  # 소수점 3자리 반올림
        
    return str(tuple(gradient_tuple))

def tuple_to_sympy(tup, local_dict):
    if isinstance(tup, int):
        return sympy.Integer(tup)
    if isinstance(tup, float):
        return sympy.Float(tup)
    if isinstance(tup, str):
        return local_dict.get(tup, sympy.Symbol(tup))
    
    op = tup[0]
    if op == 'Add':
        return tuple_to_sympy(tup[1], local_dict) + tuple_to_sympy(tup[2], local_dict)
    elif op == 'Mul':
        return tuple_to_sympy(tup[1], local_dict) * tuple_to_sympy(tup[2], local_dict)
    elif op == 'Pow':
        return tuple_to_sympy(tup[1], local_dict) ** tuple_to_sympy(tup[2], local_dict)
    elif op == 'Exp':
        return sympy.exp(tuple_to_sympy(tup[1], local_dict))
    elif op == 'Sin':
        return sympy.sin(tuple_to_sympy(tup[1], local_dict))
    elif op == 'Cos':
        return sympy.cos(tuple_to_sympy(tup[1], local_dict))
    elif op == 'Log':
        return sympy.log(tuple_to_sympy(tup[1], local_dict))
    return tup

def simplify_tuple(tup):
    if tup is None:
        return 0  # None 센티넬은 0으로 안전하게 수렴
    if not isinstance(tup, tuple):
        return tup
    op = tup[0]
    # 자식 재귀 호출 후 None이 나오면 0으로 대체
    def _s(c):
        r = simplify_tuple(c)
        return r if r is not None else 0
    args = [_s(child) for child in tup[1:]]
    if op == 'Add':
        if args[0] == 0: return args[1]
        if args[1] == 0: return args[0]
        if isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)): return args[0] + args[1]
        if args[0] == args[1]: return ('Mul', 2, args[0])
    elif op == 'Mul':
        if args[0] == 0 or args[1] == 0: return 0
        if args[0] == 1: return args[1]
        if args[1] == 1: return args[0]
        if isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)): return args[0] * args[1]
    elif op == 'Pow':
        if args[1] == 0: return 1
        if args[1] == 1: return args[0]
        if args[0] == 0: return 0
        if args[0] == 1: return 1
        if isinstance(args[0], (int, float)) and isinstance(args[1], (int, int)): return args[0] ** args[1]
    elif op == 'Exp':
        if args[0] == 0: return 1
    elif op == 'Sin':
        if args[0] == 0: return 0
        if isinstance(args[0], (int, float)): return 0
    elif op == 'Cos':
        if args[0] == 0: return 1
        if isinstance(args[0], (int, float)): return 1
    elif op == 'Log':
        if args[0] == 1: return 0
        if args[0] == 0: return 0   # log(0)=-inf → 0으로 대체 (발산 방지)
        if isinstance(args[0], (int, float)) and args[0] > 0: return args[0]
    return (op, *args)

def _has_node_type(tup, type_set):
    """튜플 트리에 특정 연산자 타입이 존재하는지 재귀 탐색"""
    if not isinstance(tup, tuple): return False
    if tup[0] in type_set: return True
    return any(_has_node_type(child, type_set) for child in tup[1:])

def _has_decay_outside_trig(tup):
    """Sin/Cos 서브트리 바깥에 감쇠 인자(Exp, Pow)가 존재하는지 확인.
    Sin(Exp(ωt)) 같은 처프 구조를 탐지하여 차단하기 위한 함수."""
    if not isinstance(tup, tuple): return False
    op = tup[0]
    # Sin/Cos 안으로는 재귀 진입 금지 — 내부의 Exp/Pow는 감쇠로 인정 안 함
    if op in TRIG_OPS: return False
    # 현재 노드가 감쇠 인자면 성공
    if op in DECAY_OPS: return True
    return any(_has_decay_outside_trig(child) for child in tup[1:])

import cmath

def _eval_ast_num(tup, vdict):
    if isinstance(tup, (int, float)): return complex(tup)
    if isinstance(tup, str): 
        if tup == 'I': return 1j
        if tup == 'pi': return cmath.pi
        return complex(vdict.get(tup, 1.0))
    if not isinstance(tup, tuple) or not tup: return complex(1.0)
    op = tup[0]
    try:
        v1 = _eval_ast_num(tup[1], vdict)
        if len(tup) > 2: v2 = _eval_ast_num(tup[2], vdict)
        if op == 'Add': return v1 + v2
        if op == 'Mul': return v1 * v2
        if op == 'Pow': return v1 ** v2
        if op == 'Exp': return cmath.exp(v1)
        if op == 'Sin': return cmath.sin(v1)
        if op == 'Cos': return cmath.cos(v1)
        if op == 'Log': return cmath.log(v1)
    except Exception:
        return complex(float('inf'), float('inf'))
    return complex(1.0)

def passes_asymptotic_constraints(tup, tmpl):
    if tup is None: return False
    
    has_exp_tmpl = False
    if tmpl is not None and tmpl.has(sympy.exp):
        has_exp_tmpl = True

    def get_val(t_val):
        vdict = {'t': t_val, 'omega': 2.3, 'a': 1.5, 'b': 0.7, 's': 1.1}
        val = _eval_ast_num(tup, vdict)
        if has_exp_tmpl:
            try: val = cmath.exp(val)
            except Exception: val = complex(float('inf'), float('inf'))
        return abs(val)

    try:
        import math
        v10 = get_val(10.0)
        v20 = get_val(20.0)
    except Exception:
        return False
        
    if math.isinf(v20) or math.isnan(v20): return False
    
    # 지수함수적 발산(Exponential Divergence)을 차단합니다.
    # 수렴, 진동(오일러 공식), 혹은 다항식적 증가(t^2 등)는 통과시킵니다.
    # v20 / v10 비율이 지나치게 크면 기각합니다. (예: e^10 = 22026)
    if v10 > 1e-7:
        growth_ratio = v20 / v10
        if growth_ratio > 100.0:
            return False
    # v10 ≈ 0 이면 진동 함수의 영점 통과 → 발산 아님, 통과

    return True

def get_free_symbols_from_tuple(tup):
    if tup is None: return set()
    if isinstance(tup, str):
        return {tup}
    elif isinstance(tup, (int, float)):
        return set()
    syms = set()
    for child in tup[1:]:
        syms.update(get_free_symbols_from_tuple(child))
    return syms

def subs_in_tuple(tup, target_str, replacement_str):
    if tup == target_str:
        return replacement_str
    if not isinstance(tup, tuple):
        return tup
    return (tup[0],) + tuple(subs_in_tuple(child, target_str, replacement_str) for child in tup[1:])

def prevents_inverse_cancellation(kernel_expr, chosen_template, unknown_sym):
    import sympy
    try:
        test_expr = chosen_template.subs(unknown_sym, kernel_expr)
    except:
        return False
        
    f_t_sym = sympy.Symbol('f_t')
    # 대수형(Algebraic) 꼼수 방지: f_t가 완전히 소거되었는지 검사
    try:
        # simplify는 무한루프 가능성이 높으므로 가장 빠른 cancel/expand만 사용
        simp_expr = sympy.cancel(test_expr)
        if not simp_expr.has(f_t_sym):
            return False
    except:
        pass # 실패시 엄격한 검사 보류

    template_funcs = {type(f) for f in chosen_template.atoms(sympy.Function) if unknown_sym in f.free_symbols}
    if not template_funcs:
        return True
        
    for func_type in template_funcs:
        if not test_expr.has(func_type):
            return False
    return True

# --- END TUPLE AST FUNCTIONS ---

laguerre_nodes, laguerre_weights = np.polynomial.laguerre.laggauss(50)
hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(50)
legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(150)

def evaluate_integral_fitness(operator_expr, dataset) -> float:
    score = 0
    import random
    
    if not isinstance(operator_expr, Integral):
        return 0.0
        
    integrand_base = operator_expr.args[0]
    limits = operator_expr.args[1]
    # indefinite Integral은 limits가 Symbol → isinstance 체크로 TypeError 완전 차단
    if not isinstance(limits, (tuple, sympy.Tuple)) or len(limits) < 3:
        return 0.0
    int_var = limits[0]
    lower = limits[1]
    upper = limits[2]
    
    f_t_sym = Symbol('f_t')
    
    quad_type = 'legendre'
    if lower == 0 and upper == oo:
        quad_type = 'laguerre'
    elif lower == -oo and upper == oo:
        quad_type = 'hermite'

    if not dataset: return 0.0
    
    # [수정 1] 모든 데이터행에서 파라미터 수집 (특정 행에 편향되지 않는 범용 색인)
    target_syms_set = set()
    for row in dataset:
        target_syms_set.update(row['target_expr'].free_symbols)
    target_syms = list({s for s in target_syms_set if s.name not in ['pi', 'E', 'I', int_var.name, f_t_sym.name]})
    
    # ── [소볼레프] 편미분 기준 변수 자동 선택 ────────────────────────────────────
    # 1순위: injected 변수(입력엔 없고 출력에 등장 → 연산자 도메인 변수, e.g. s)
    # 2순위: retained 변수(입력/출력 모두 존재 → 물리 파라미터, e.g. a, w)
    in_syms_set = set()
    for row in dataset:
        in_syms_set.update(row['input_expr'].free_symbols)
    in_syms_names = {s.name for s in in_syms_set if s.name not in ['pi', 'E', 'I', int_var.name, f_t_sym.name]}
    out_syms_names = {s.name for s in target_syms}
    injected_names = out_syms_names - in_syms_names  # 출력에만 있는 것 (s, w ...)
    retained_names = out_syms_names & in_syms_names  # 공통 (a, b ...)
    
    diff_priority = injected_names if injected_names else retained_names
    diff_sym = next((s for s in target_syms if s.name in diff_priority), None)
    
    # ── [소볼레프] 후보 커널의 도함수 lambdify ────────────────────────────────────
    f_num_grad = None  # 도함수 계산 실패 시 None으로 값 채점만 사용
    if diff_sym is not None:
        try:
            integrand_deriv = sympy.diff(integrand_base, diff_sym)
            f_num_grad = lambdify([int_var, f_t_sym] + target_syms, integrand_deriv, modules=['numpy', 'scipy'])
        except Exception:
            f_num_grad = None

    try:
        f_num = lambdify([int_var, f_t_sym] + target_syms, integrand_base, modules=['numpy', 'scipy'])
    except Exception:
        return 0.0

    for row in dataset:
        if 'f_in_lambdified' not in row:
            row['f_in_lambdified'] = lambdify([int_var] + target_syms, row['input_expr'], modules=['numpy', 'scipy'])
        if 'f_target_lambdified' not in row:
            row['f_target_lambdified'] = lambdify(target_syms, row['target_expr'], modules=['numpy', 'scipy'])
            
        # ── [소볼레프] 타겟 도함수 lambdify (행별 1회 캐싱) ──────────────────────
        if 'f_target_deriv_lambdified' not in row and diff_sym is not None:
            try:
                target_deriv_expr = sympy.diff(row['target_expr'], diff_sym)
                row['f_target_deriv_lambdified'] = lambdify(target_syms, target_deriv_expr, modules=['numpy', 'scipy'])
            except Exception:
                row['f_target_deriv_lambdified'] = None
            
        f_in = row['f_in_lambdified']
        f_target = row['f_target_lambdified']
        f_target_deriv = row.get('f_target_deriv_lambdified')
            
        if 'test_points' not in row:
            test_points = []
            for _ in range(3):
                pt = None
                for _attempt in range(100):
                    pt = {}
                    for sym in target_syms:
                        # 심볼의 물리적 속성 존중: positive로 선언된 경우 양수만 생성
                        if sym.is_positive or sym.is_nonnegative:
                            val = 10 ** random.uniform(-0.5, 0.4)
                        else:
                            sign = random.choice([-1.0, 1.0])
                            val = sign * (10 ** random.uniform(-0.5, 0.4))
                        pt[sym.name] = val
                    
                    if row.get('roc') is not None:
                        subs_dict = {sym: pt[sym.name] for sym in target_syms}
                        roc_val = row['roc']
                        if isinstance(roc_val, bool):
                            if roc_val: break
                        else:
                            if roc_val.subs(subs_dict) == True:
                                break
                    else:
                        break
                test_points.append(pt)
            row['test_points'] = test_points
            
        test_points = row['test_points']
            
        mse_sum = 0
        valid_points = 0
        
        for pt in test_points:
            target_args = [pt[sym.name] for sym in target_syms]
            try:
                tv = f_target(*target_args)
                if np.isnan(tv) or np.isinf(tv): continue
                
                with np.errstate(all='ignore'):
                    if quad_type == 'laguerre':
                        u = 0.5 * (legendre_nodes + 1)
                        du_weights = 0.5 * legendre_weights
                        u = np.clip(u, 0.0, 1.0 - 1e-12)
                        x = u / (1.0 - u)
                        jacobian = 1.0 / ((1.0 - u)**2)
                        f_t_vals = np.nan_to_num(np.asarray(f_in(x, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        y = np.nan_to_num(np.asarray(f_num(x, f_t_vals, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        calc_val = np.sum(du_weights * jacobian * y)
                    elif quad_type == 'hermite':
                        u = 0.5 * (legendre_nodes + 1)
                        du_weights = 0.5 * legendre_weights
                        u = np.clip(u, 0.0, 1.0 - 1e-12)
                        jac = 1.0 / ((1.0 - u)**2)
                        x_pos = u / (1.0 - u)
                        f_t_pos = np.nan_to_num(np.asarray(f_in(x_pos, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        y_pos = np.nan_to_num(np.asarray(f_num(x_pos, f_t_pos, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        x_neg = -u / (1.0 - u)
                        f_t_neg = np.nan_to_num(np.asarray(f_in(x_neg, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        y_neg = np.nan_to_num(np.asarray(f_num(x_neg, f_t_neg, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        calc_val = np.sum(du_weights * jac * (y_pos + y_neg))
                    else:
                        # [domain 지원] 심볼릭 구간(e.g. -D/2, D/2) → 파라미터 수치 치환 후 적분
                        try:
                            subs_d = {sym: pt[sym.name] for sym in target_syms if sym.name in pt}
                            a_val = float(lower.subs(subs_d).evalf()) if not lower.is_number else float(lower.evalf())
                            b_val = float(upper.subs(subs_d).evalf()) if not upper.is_number else float(upper.evalf())
                        except Exception:
                            a_val, b_val = 0.0, 10.0
                        # Gauss-Legendre on [a_val, b_val]: nodes [-1,1] → [a_val, b_val]
                        x_fin = 0.5 * (b_val - a_val) * legendre_nodes + 0.5 * (a_val + b_val)
                        jac_fin = 0.5 * (b_val - a_val)
                        f_t_fin = np.nan_to_num(np.asarray(f_in(x_fin, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        y_fin   = np.nan_to_num(np.asarray(f_num(x_fin, f_t_fin, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        calc_val = jac_fin * np.sum(legendre_weights * y_fin)
                        # 도함수 계산을 위해 outer scope 변수에 저장
                        _x_fin_outer = x_fin
                        _jac_fin_outer = jac_fin
                        _f_t_fin_outer = f_t_fin
                        
                if np.isnan(calc_val) or np.isinf(calc_val) or (np.iscomplexobj(calc_val) and abs(calc_val.imag) > 1e-4):
                    mse = 1e6
                else:
                    if np.iscomplexobj(calc_val): calc_val = calc_val.real
                    tv_f = float(tv)
                    mse_val = (tv_f - calc_val) ** 2
                    
                    # ── [소볼레프] 도함수 MSE 추가 검증 ──────────────────────────────
                    mse_grad = 0.0
                    if f_num_grad is not None and f_target_deriv is not None:
                        try:
                            tv_deriv = float(f_target_deriv(*target_args))
                            if not (np.isnan(tv_deriv) or np.isinf(tv_deriv)):
                                with np.errstate(all='ignore'):
                                    if quad_type == 'laguerre':
                                        y_grad = np.asarray(f_num_grad(x, f_t_vals, *target_args))
                                        calc_deriv = np.sum(du_weights * jacobian * y_grad)
                                    elif quad_type == 'hermite':
                                        y_grad = np.asarray(f_num_grad(x, f_t_vals, *target_args))
                                        calc_deriv = np.sum(hermite_weights * y_grad * np.exp(x**2))
                                    else:
                                        # 유한 구간 도함수: 동일한 x_fin/jac_fin으로 GL 적분
                                        try:
                                            y_grad = np.asarray(f_num_grad(_x_fin_outer, _f_t_fin_outer, *target_args))
                                            calc_deriv = _jac_fin_outer * np.sum(legendre_weights * y_grad)
                                        except Exception:
                                            calc_deriv = 0.0
                                if not (np.isnan(calc_deriv) or np.isinf(calc_deriv)):
                                    if np.iscomplexobj(calc_deriv): calc_deriv = calc_deriv.real
                                    mse_grad = (tv_deriv - float(calc_deriv)) ** 2
                                else:
                                    mse_grad = 1e6
                        except Exception:
                            mse_grad = 0.0  # 도함수 계산 실패 시 값 채점만 사용
                            
                    mse = mse_val + mse_grad
                    
            except Exception:
                mse = 1e6
                
            mse_sum += mse
            valid_points += 1
            
        if valid_points == 0:
            return 0.0
            
        avg_mse = mse_sum / valid_points
        res_score = 10.0 / (1.0 + avg_mse)
        score += res_score
        
        if res_score < 0.1:
            return (score / (10.0 * len(dataset))) * 100.0
            
    return (score / (10.0 * len(dataset))) * 100.0

def evaluate_integral_complex_fitness(operator_expr, dataset) -> float:
    score = 0
    import random
    
    if not isinstance(operator_expr, Integral):
        return 0.0
        
    integrand_base = operator_expr.args[0]
    limits = operator_expr.args[1]
    if not isinstance(limits, (tuple, sympy.Tuple)) or len(limits) < 3:
        return 0.0
    int_var = limits[0]
    lower = limits[1]
    upper = limits[2]
    
    f_t_sym = Symbol('f_t')
    
    quad_type = 'legendre'
    if lower == 0 and upper == oo:
        quad_type = 'laguerre'
    elif lower == -oo and upper == oo:
        quad_type = 'hermite'

    if not dataset: return 0.0
    
    # [수정 1] 모든 데이터행에서 파라미터 수집 (특정 행에 편향되지 않는 범용 색인)
    target_syms_set = set()
    for row in dataset:
        target_syms_set.update(row['target_expr'].free_symbols)
    target_syms = list({s for s in target_syms_set if s.name not in ['pi', 'E', 'I', int_var.name, f_t_sym.name]})
    
    # ── [소볼레프] 편미분 기준 변수 자동 선택 ────────────────────────────────────
    # 1순위: injected 변수(입력엔 없고 출력에 등장 → 연산자 도메인 변수, e.g. s)
    # 2순위: retained 변수(입력/출력 모두 존재 → 물리 파라미터, e.g. a, w)
    in_syms_set = set()
    for row in dataset:
        in_syms_set.update(row['input_expr'].free_symbols)
    in_syms_names = {s.name for s in in_syms_set if s.name not in ['pi', 'E', 'I', int_var.name, f_t_sym.name]}
    out_syms_names = {s.name for s in target_syms}
    injected_names = out_syms_names - in_syms_names  # 출력에만 있는 것 (s, w ...)
    retained_names = out_syms_names & in_syms_names  # 공통 (a, b ...)
    
    diff_priority = injected_names if injected_names else retained_names
    diff_sym = next((s for s in target_syms if s.name in diff_priority), None)
    
    # ── [소볼레프] 후보 커널의 도함수 lambdify ────────────────────────────────────
    f_num_grad = None  # 도함수 계산 실패 시 None으로 값 채점만 사용
    if diff_sym is not None:
        try:
            integrand_deriv = sympy.diff(integrand_base, diff_sym)
            f_num_grad = lambdify([int_var, f_t_sym] + target_syms, integrand_deriv, modules=['numpy', 'scipy'])
        except Exception:
            f_num_grad = None

    try:
        f_num = lambdify([int_var, f_t_sym] + target_syms, integrand_base, modules=['numpy', 'scipy'])
    except Exception:
        return 0.0

    for row in dataset:
        if 'f_in_lambdified' not in row:
            row['f_in_lambdified'] = lambdify([int_var] + target_syms, row['input_expr'], modules=['numpy', 'scipy'])
        if 'f_target_lambdified' not in row:
            row['f_target_lambdified'] = lambdify(target_syms, row['target_expr'], modules=['numpy', 'scipy'])
            
        # ── [소볼레프] 타겟 도함수 lambdify (행별 1회 캐싱) ──────────────────────
        if 'f_target_deriv_lambdified' not in row and diff_sym is not None:
            try:
                target_deriv_expr = sympy.diff(row['target_expr'], diff_sym)
                row['f_target_deriv_lambdified'] = lambdify(target_syms, target_deriv_expr, modules=['numpy', 'scipy'])
            except Exception:
                row['f_target_deriv_lambdified'] = None
            
        f_in = row['f_in_lambdified']
        f_target = row['f_target_lambdified']
        f_target_deriv = row.get('f_target_deriv_lambdified')
            
        if 'test_points' not in row:
            test_points = []
            for _ in range(3):
                pt = None
                for _attempt in range(100):
                    pt = {}
                    for sym in target_syms:
                        # 심볼의 물리적 속성 존중: positive로 선언된 경우 양수만 생성
                        if sym.is_positive or sym.is_nonnegative:
                            val = 10 ** random.uniform(-0.5, 0.4)
                        else:
                            sign = random.choice([-1.0, 1.0])
                            val = sign * (10 ** random.uniform(-0.5, 0.4))
                        pt[sym.name] = val
                    
                    if row.get('roc') is not None:
                        subs_dict = {sym: pt[sym.name] for sym in target_syms}
                        roc_val = row['roc']
                        if isinstance(roc_val, bool):
                            if roc_val: break
                        else:
                            if roc_val.subs(subs_dict) == True:
                                break
                    else:
                        break
                test_points.append(pt)
            row['test_points'] = test_points
            
        test_points = row['test_points']
            
        mse_sum = 0
        valid_points = 0
        
        for pt in test_points:
            target_args = [pt[sym.name] for sym in target_syms]
            try:
                tv = f_target(*target_args)
                if np.isnan(tv) or np.isinf(tv): continue
                
                with np.errstate(all='ignore'):
                    if quad_type == 'laguerre':
                        u = 0.5 * (legendre_nodes + 1)
                        du_weights = 0.5 * legendre_weights
                        u = np.clip(u, 0.0, 1.0 - 1e-12)
                        x = u / (1.0 - u)
                        jacobian = 1.0 / ((1.0 - u)**2)
                        f_t_vals = np.nan_to_num(np.asarray(f_in(x, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        y = np.nan_to_num(np.asarray(f_num(x, f_t_vals, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        calc_val = np.sum(du_weights * jacobian * y)
                    elif quad_type == 'hermite':
                        u = 0.5 * (legendre_nodes + 1)
                        du_weights = 0.5 * legendre_weights
                        u = np.clip(u, 0.0, 1.0 - 1e-12)
                        jac = 1.0 / ((1.0 - u)**2)
                        x_pos = u / (1.0 - u)
                        f_t_pos = np.nan_to_num(np.asarray(f_in(x_pos, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        y_pos = np.nan_to_num(np.asarray(f_num(x_pos, f_t_pos, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        x_neg = -u / (1.0 - u)
                        f_t_neg = np.nan_to_num(np.asarray(f_in(x_neg, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        y_neg = np.nan_to_num(np.asarray(f_num(x_neg, f_t_neg, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        calc_val = np.sum(du_weights * jac * (y_pos + y_neg))
                    else:
                        # [domain 지원] 심볼릭 구간 → 파라미터 수치 치환 후 적분
                        try:
                            subs_d = {sym: pt[sym.name] for sym in target_syms if sym.name in pt}
                            a_val = float(lower.subs(subs_d).evalf()) if not lower.is_number else float(lower.evalf())
                            b_val = float(upper.subs(subs_d).evalf()) if not upper.is_number else float(upper.evalf())
                        except Exception:
                            a_val, b_val = 0.0, 10.0
                        x_fin = 0.5 * (b_val - a_val) * legendre_nodes + 0.5 * (a_val + b_val)
                        jac_fin = 0.5 * (b_val - a_val)
                        f_t_fin = np.nan_to_num(np.asarray(f_in(x_fin, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        y_fin   = np.nan_to_num(np.asarray(f_num(x_fin, f_t_fin, *target_args)), nan=0.0, posinf=0.0, neginf=0.0)
                        calc_val = jac_fin * np.sum(legendre_weights * y_fin)
                        _x_fin_outer = x_fin
                        _jac_fin_outer = jac_fin
                        _f_t_fin_outer = f_t_fin
                        
                if np.isnan(calc_val) or np.isinf(calc_val):
                    mse = 1e6
                else:
                    mse_val = abs(complex(tv) - complex(calc_val)) ** 2
                    
                    # ── [소볼레프] 도함수 MSE 추가 검증 ──────────────────────────────
                    mse_grad = 0.0
                    if f_num_grad is not None and f_target_deriv is not None:
                        try:
                            tv_deriv = float(f_target_deriv(*target_args))
                            if not (np.isnan(tv_deriv) or np.isinf(tv_deriv)):
                                with np.errstate(all='ignore'):
                                    if quad_type == 'laguerre':
                                        y_grad = np.asarray(f_num_grad(x, f_t_vals, *target_args))
                                        calc_deriv = np.sum(du_weights * jacobian * y_grad)
                                    elif quad_type == 'hermite':
                                        y_grad = np.asarray(f_num_grad(x, f_t_vals, *target_args))
                                        calc_deriv = np.sum(hermite_weights * y_grad * np.exp(x**2))
                                    else:
                                        try:
                                            y_grad = np.asarray(f_num_grad(_x_fin_outer, _f_t_fin_outer, *target_args))
                                            calc_deriv = _jac_fin_outer * np.sum(legendre_weights * y_grad)
                                        except Exception:
                                            calc_deriv = 0.0
                                if not (np.isnan(calc_deriv) or np.isinf(calc_deriv)):
                                    mse_grad = abs(complex(tv_deriv) - complex(calc_deriv)) ** 2
                                else:
                                    mse_grad = 1e6
                        except Exception:
                            mse_grad = 0.0  # 도함수 계산 실패 시 값 채점만 사용
                            
                    mse = mse_val + mse_grad
                    
            except Exception:
                mse = 1e6
                
            mse_sum += mse
            valid_points += 1
            
        if valid_points == 0:
            return 0.0
            
        avg_mse = mse_sum / valid_points
        res_score = 10.0 / (1.0 + avg_mse)
        score += res_score
        
        if res_score < 0.1:
            return (score / (10.0 * len(dataset))) * 100.0
            
    return (score / (10.0 * len(dataset))) * 100.0

def evaluate_algebraic_fitness(operator_expr, dataset) -> float:
    score = 0.0
    import random
    
    # 템플릿이 Integral/Derivative가 아니면 Algebraic으로 간주
    if isinstance(operator_expr, (Integral, Derivative)):
        return 0.0
        
    f_t_sym = sympy.Symbol('f_t')
    
    # 파라미터 수집 (타겟, 입력 모두)
    all_syms_set = set()
    for row in dataset:
        all_syms_set.update(row['target_expr'].free_symbols)
        all_syms_set.update(row['input_expr'].free_symbols)
    
    all_syms = list({s for s in all_syms_set if s.name not in ['pi', 'E', 'I', f_t_sym.name]})
    
    try:
        f_operator_lambdified = lambdify([f_t_sym] + all_syms, operator_expr, modules=['numpy', 'scipy'])
    except Exception:
        return 0.0

    for row in dataset:
        if 'f_target_lambdified_alg' not in row:
            row['f_target_lambdified_alg'] = lambdify(all_syms, row['target_expr'], modules=['numpy', 'scipy'])
        if 'f_in_lambdified_alg' not in row:
            row['f_in_lambdified_alg'] = lambdify(all_syms, row['input_expr'], modules=['numpy', 'scipy'])
            
        f_target = row['f_target_lambdified_alg']
        f_in = row['f_in_lambdified_alg']
            
        if 'test_points_alg' not in row:
            test_points = []
            for _ in range(5):
                pt = {}
                for sym in all_syms:
                    sign = random.choice([-1.0, 1.0])
                    val = sign * (10 ** random.uniform(-0.5, 0.4))
                    pt[sym.name] = val
                test_points.append(pt)
            row['test_points_alg'] = test_points
            
        test_points = row['test_points_alg']
            
        mse_sum = 0
        valid_points = 0
        
        for pt in test_points:
            target_args = [pt[sym.name] for sym in all_syms]
            try:
                tv = f_target(*target_args)
                in_v = f_in(*target_args)
                
                if np.isnan(tv) or np.isinf(tv) or np.isnan(in_v) or np.isinf(in_v): continue
                
                calc_val = f_operator_lambdified(in_v, *target_args)
                
                if np.isnan(calc_val) or np.isinf(calc_val):
                    mse = 1e6
                else:
                    if np.iscomplexobj(tv) or np.iscomplexobj(calc_val):
                        mse = abs(complex(tv) - complex(calc_val)) ** 2
                    else:
                        mse = (float(tv) - float(calc_val)) ** 2
                        
            except Exception:
                mse = 1e6
                
            mse_sum += mse
            valid_points += 1
            
        if valid_points == 0:
            return 0.0
            
        avg_mse = mse_sum / valid_points
        res_score = 10.0 / (1.0 + avg_mse)
        score += res_score
        
        if res_score < 0.1:
            return (score / (10.0 * len(dataset))) * 100.0
            
    return (score / (10.0 * len(dataset))) * 100.0

def evaluate_derivative_fitness(operator_expr, dataset) -> float:
    import numpy as np
    import sympy
    from sympy import lambdify
    
    score = 0.0
    f_t_sym = sympy.Symbol('f_t')
    
    for row in dataset:
        # [Bug 2 수정] input_expr + target_expr 양쪽에서 자유 변수 수집
        # 미분 결과에는 미분 변수(t 등)가 잔존하므로 input_expr 쪽도 반드시 포함
        if 'input_expr' not in row:
            return 0.0
        all_syms_set = row['target_expr'].free_symbols | row['input_expr'].free_symbols
        all_syms = list({s for s in all_syms_set if s.name not in ['pi', 'E', 'I', f_t_sym.name]})
        
        # [Bug 2 수정] 미분형 전용 캐시 키 사용 (적분형 test_points 오염 방지)
        if 'test_points_deriv' not in row:
            test_pts = []
            for _ in range(5):
                pt = {}
                for sym in all_syms:
                    pt[sym.name] = random.choice([-1.0, 1.0]) * (10 ** random.uniform(-0.5, 0.4))
                test_pts.append(pt)
            row['test_points_deriv'] = test_pts
        test_points = row['test_points_deriv']
        
        try:
            # 1. 기호 함수 치환
            analytic_operator = operator_expr.subs(f_t_sym, row['input_expr'])
            # 2. 해석적 미분 실행 (doit)
            analytic_operator = analytic_operator.doit()
        except Exception:
            return 0.0
            
        try:
            f_operator_lambdified = lambdify(all_syms, analytic_operator, modules=['numpy', 'scipy'])
            f_target = lambdify(all_syms, row['target_expr'], modules=['numpy', 'scipy'])
        except Exception:
            return 0.0
            
        mse_sum = 0
        valid_points = 0
        
        for pt in test_points:
            target_args = [pt[sym.name] for sym in all_syms]
            try:
                tv = f_target(*target_args)
                calc_val = f_operator_lambdified(*target_args)
                
                if np.isnan(tv) or np.isinf(tv) or np.isnan(calc_val) or np.isinf(calc_val):
                    mse = 1e6
                else:
                    if np.iscomplexobj(tv) or np.iscomplexobj(calc_val):
                        mse = abs(complex(tv) - complex(calc_val)) ** 2
                    else:
                        mse = (float(tv) - float(calc_val)) ** 2
            except Exception:
                mse = 1e6
                
            mse_sum += mse
            valid_points += 1
            
        if valid_points == 0:
            return 0.0
            
        avg_mse = mse_sum / valid_points
        res_score = 10.0 / (1.0 + avg_mse)
        score += res_score
        
        if res_score < 0.1:
            return (score / (10.0 * len(dataset))) * 100.0
            
    return (score / (10.0 * len(dataset))) * 100.0

def dispatch_fitness(operator_expr, dataset, is_complex_domain) -> float:
    # 1. 미분형
    if isinstance(operator_expr, Derivative) or operator_expr.has(Derivative):
        return evaluate_derivative_fitness(operator_expr, dataset)
    # 2. 적분형
    elif isinstance(operator_expr, Integral):
        if is_complex_domain:
            return evaluate_integral_complex_fitness(operator_expr, dataset)
        else:
            return evaluate_integral_fitness(operator_expr, dataset)
    # 3. 대수형 (나머지 전부)
    else:
        return evaluate_algebraic_fitness(operator_expr, dataset)

class AxiomaticDeductiveEngine:
    def __init__(self):
        self.f_t = Symbol('f_t')
        self.unknown_sym = Symbol('Unknown_Term')
        
    def deduce_operator_axioms(self, dataset) -> dict:
        import sympy
        
        # 공리 분석 결과: 가능성을 모두 True로 열어두고, 모순 발견 시 False로 닫습니다.
        axioms = {
            'can_be_algebraic': True,
            'can_be_derivative_1': True,
            'can_be_derivative_2': True
        }
        
        for row in dataset:
            in_expr = row.get('input_expr')
            out_expr = row.get('target_expr')
            if not in_expr or not out_expr:
                continue
                
            in_terms = in_expr.as_ordered_terms() if hasattr(in_expr, 'as_ordered_terms') else [in_expr]
            out_terms = out_expr.as_ordered_terms() if hasattr(out_expr, 'as_ordered_terms') else [out_expr]
            
            # 1. 라이프니츠 공리 (항의 팽창)
            # 입력은 단일항인데 출력이 다중항이면 대수형 불가능
            if len(in_terms) == 1 and len(out_terms) > 1:
                axioms['can_be_algebraic'] = False
                
            # 단일항 대 단일항 비교
            if len(in_terms) == 1 and len(out_terms) == 1:
                in_t = in_terms[0]
                out_t = out_terms[0]
                
                # 2. 조화 함수 위상 편이 공리
                if in_t.has(sympy.sin) or in_t.has(sympy.cos):
                    # 입력에 sin이 있고, 출력에 cos만 남았다면 (위상 변화) -> 1계 미분
                    if in_t.has(sympy.sin) and out_t.has(sympy.cos) and not out_t.has(sympy.sin):
                        axioms['can_be_algebraic'] = False
                        axioms['can_be_derivative_2'] = False
                    # 입력에 cos이 있고, 출력에 sin만 남았다면 (위상 변화) -> 1계 미분
                    elif in_t.has(sympy.cos) and out_t.has(sympy.sin) and not out_t.has(sympy.cos):
                        axioms['can_be_algebraic'] = False
                        axioms['can_be_derivative_2'] = False
                    # 입력에 sin이 있고 출력에도 sin이 있다면 -> 짝수 차수 (대수형 or 2계)
                    elif in_t.has(sympy.sin) and out_t.has(sympy.sin) and not out_t.has(sympy.cos):
                        axioms['can_be_derivative_1'] = False
                        
                # 3. 다항식 차수 강하 공리
                try:
                    free_vars = in_t.free_symbols
                    if free_vars:
                        var = list(free_vars)[0]
                        if in_t.is_polynomial(var) and out_t.is_polynomial(var):
                            in_deg = sympy.degree(in_t, gen=var)
                            out_deg = sympy.degree(out_t, gen=var)
                            
                            if in_deg > 0:
                                deg_diff = in_deg - out_deg
                                if deg_diff == 0:
                                    axioms['can_be_derivative_1'] = False
                                    axioms['can_be_derivative_2'] = False
                                elif deg_diff == 1:
                                    axioms['can_be_algebraic'] = False
                                    axioms['can_be_derivative_2'] = False
                                elif deg_diff == 2:
                                    axioms['can_be_algebraic'] = False
                                    axioms['can_be_derivative_1'] = False
                except Exception:
                    pass
                    
        return axioms

    def analyze_dataset(self, dataset: List[Dict[str, Any]], domain_bounds=None):
        all_in_syms = set()
        all_out_syms = set()
        
        for row in dataset:
            in_expr = row.get('input_expr')
            out_expr = row.get('target_expr')
            if in_expr:
                all_in_syms.update(in_expr.free_symbols)
                if in_expr.has(sympy.I): all_in_syms.add(sympy.I)
                if in_expr.has(sympy.pi): all_in_syms.add(sympy.pi)
                if in_expr.has(sympy.E): all_in_syms.add(sympy.E)
            if out_expr:
                all_out_syms.update(out_expr.free_symbols)
                if out_expr.has(sympy.I): all_out_syms.add(sympy.I)
                if out_expr.has(sympy.pi): all_out_syms.add(sympy.pi)
                if out_expr.has(sympy.E): all_out_syms.add(sympy.E)
                
        raw_symbols = (all_in_syms.union(all_out_syms)) - {self.f_t, self.unknown_sym}
        
        i_must_be_in_kernel = False
        for row in dataset:
            in_expr = row.get('input_expr')
            out_expr = row.get('target_expr')
            in_has_i = in_expr and in_expr.has(sympy.I)
            out_has_i = out_expr and out_expr.has(sympy.I)
            
            # 만약 단 하나의 행이라도 실수->복소수, 혹은 복소수->실수로 차원 전이가 일어난다면
            # 미지 커널(연산자)은 반드시 I를 조작하는 능력이 있어야 합니다.
            if in_has_i != out_has_i:
                i_must_be_in_kernel = True
                break
                
        ignore_syms = {self.f_t, self.unknown_sym, sympy.E, sympy.pi, 
                       c_light, g_grav, G_grav, k_B, h_bar, Z_0, mu_0, eps_0}
                       
        if not i_must_be_in_kernel:
            ignore_syms.add(sympy.I)
        else:
            # I가 전역 교집합 로직에 의해 누락되는 것을 막기 위해,
            # 만약 주입되어야 한다면 out_syms에 확실히 있도록 보장하고 in_syms에서는 제거
            # (차원 전이가 확인된 이상 I는 명확한 주입/소거 대상)
            if sympy.I in all_in_syms and sympy.I in all_out_syms:
                all_in_syms.discard(sympy.I)
        all_in_syms = {sym for sym in all_in_syms if sym not in ignore_syms}
        all_out_syms = {sym for sym in all_out_syms if sym not in ignore_syms}
        
        eliminated = list(all_in_syms - all_out_syms)
        injected = list(all_out_syms - all_in_syms)
        retained = list(all_in_syms & all_out_syms)
        
        base_template = self.f_t * self.unknown_sym
        templates = []
        
        # ── 입력 함수 파라미터 vs 진짜 적분변수 구별 ──────────────────────────────
        # 핵심 판별: eliminated 변수가 출력 표현식에도 나타나면 그것은 '입력 함수 파라미터'
        # (예: sin(omega*t)→a*sin(omega*t)/(b²+sin²(omega*t)), omega는 both in/out)
        # 그런 변수는 적분 변수가 아니라 대수 연산자의 파라미터임을 의미함
        # all_out_syms는 ignore_syms를 제외한 출력 자유변수이므로 직접 원시 출력에서 재확인
        raw_out_sym_names = set()
        for _row in dataset:
            _out = _row.get('target_expr')
            if _out is not None:
                raw_out_sym_names.update(s.name for s in _out.free_symbols)
        
        # 진짜 적분 변수: eliminated 중 출력에 전혀 나타나지 않는 것
        true_integral_vars = [v for v in eliminated if v.name not in raw_out_sym_names]
        # 입력 함수 파라미터: eliminated 중 출력에 나타나는 것 (대수 파라미터)
        algebraic_param_vars = [v for v in eliminated if v.name in raw_out_sym_names]
        
        if true_integral_vars:
            # 진짜 적분형 — 이 변수들만 적분 변수로 사용
            for var in true_integral_vars:
                if domain_bounds is not None:
                    lower_b, upper_b = domain_bounds
                    templates.append(Integral(base_template, (var, lower_b, upper_b)))
                    if sympy.I in raw_symbols:
                        templates.append(Integral(self.f_t * sympy.exp(self.unknown_sym), (var, lower_b, upper_b)))
                else:
                    templates.append(Integral(base_template, (var, 0, oo)))
                    templates.append(Integral(base_template, (var, -oo, oo)))
                    if sympy.I in raw_symbols:
                        templates.append(Integral(self.f_t * sympy.exp(self.unknown_sym), (var, 0, oo)))
                        templates.append(Integral(self.f_t * sympy.exp(self.unknown_sym), (var, -oo, oo)))
        
        # 대수형 또는 미분형 템플릿: retained이 있거나 algebraic_param_vars만 있는 경우
        # (대수형: eliminated 전부 algebraic_param이거나, eliminated=[] 인 경우)
        if retained or (algebraic_param_vars and not true_integral_vars):
            # 대수형 (도메인 소거 없음)
            templates.append(base_template)
            templates.append(self.f_t + self.unknown_sym)  # 덧셈 형태도 탐색
            
            # 미분형 템플릿: retained 또는 algebraic_param_vars 중에서 탐색
            diff_vars = retained if retained else algebraic_param_vars
            for var in diff_vars:
                # 1계 미분
                templates.append(self.unknown_sym * Derivative(self.f_t, var))
                templates.append(Derivative(self.unknown_sym * self.f_t, var))
                templates.append(Derivative(self.f_t, var) + self.unknown_sym)
                
                # 2계 미분
                templates.append(self.unknown_sym * Derivative(self.f_t, var, 2))
                templates.append(Derivative(self.unknown_sym * self.f_t, var, 2))
                templates.append(Derivative(self.f_t, var, 2) + self.unknown_sym)
        
        # 진짜 적분형이 있는데 algebraic_param도 있는 경우 → 대수형도 병행 탐색
        if true_integral_vars and (retained or algebraic_param_vars):
            # 두 유형 모두 탐색: 이미 적분 템플릿은 추가됐으므로 대수형도 추가
            if base_template not in templates:
                templates.append(base_template)

        if not templates:
            templates.append(base_template)
        
        # ── eliminated 재정의: 진짜 적분변수만으로 재설정 (GA 변수 탐색 범위 교정) ──
        # algebraic_param_vars는 retained 처럼 취급 (GA가 대수 탐색할 때 활용)
        effective_eliminated = true_integral_vars
        effective_retained = list(set(retained) | set(algebraic_param_vars))
        eliminated = effective_eliminated
        retained = effective_retained
            
        is_complex_domain = sympy.I in raw_symbols
        
        # ─── 공리 기반 템플릿 사전 필터링 (Axiomatic Pruning) ───
        axioms = self.deduce_operator_axioms(dataset)
        pruned_templates = []
        for tmpl in templates:
            if tmpl.has(Integral):
                pruned_templates.append(tmpl) # 적분형은 보존
            elif tmpl.has(Derivative):
                # 2계 미분인지 확인
                is_2nd_order = False
                for d in tmpl.atoms(Derivative):
                    if len(d.variables) == 2 or (len(d.variables) == 1 and hasattr(d, 'derivative_count') and d.derivative_count == 2):
                        is_2nd_order = True
                        break
                
                if is_2nd_order and axioms['can_be_derivative_2']:
                    pruned_templates.append(tmpl)
                elif not is_2nd_order and axioms['can_be_derivative_1']:
                    pruned_templates.append(tmpl)
            else:
                # 대수형
                if axioms['can_be_algebraic']:
                    pruned_templates.append(tmpl)
                    
        # Graceful Degradation: 만약 모두 모순으로 지워졌다면(불확실성), 원본을 그대로 씁니다.
        if pruned_templates:
            templates = pruned_templates

        return templates, self.unknown_sym, injected, eliminated, raw_symbols, is_complex_domain


# ── Beam Search: 얕은 트리의 잎 하나를 한 단계 확장 ─────────────────────────────
def beam_expand_kernel(kernel, terminal_pool: list, available_vars: list):
    """터미널 리프 하나를 무작위로 골라 새로운 연산자 노드로 교체(한 레벨 확장)."""
    if not isinstance(kernel, tuple):
        # 전체가 터미널이면 바로 연산자로 감싸기
        new_node_type = random.choices([1, 2, 3, 4, 5, 6], weights=[2, 4, 1, 1, 1, 1])[0]
        leaf = random.choice(terminal_pool)
        if new_node_type == 1:
            return ('Add', kernel, leaf)
        elif new_node_type == 2:
            return ('Mul', kernel, leaf)
        elif new_node_type == 3:
            return ('Pow', kernel, random.choice([-1, 1, 2]))
        elif new_node_type == 4:
            return ('Exp', kernel)
        elif new_node_type == 5:
            return ('Sin', kernel)
        elif new_node_type == 6:
            return ('Cos', kernel)

    # 트리의 모든 경로 수집
    all_paths = get_all_nodes_with_paths(kernel)
    # 터미널(잎) 경로만 추출
    leaf_paths = [(path, node) for path, ntype, node in all_paths
                  if not isinstance(node, tuple)]

    if not leaf_paths:
        return kernel

    path, leaf_node = random.choice(leaf_paths)
    leaf_b = random.choice(terminal_pool)

    new_node_type = random.choices([1, 2, 3, 4, 5, 6], weights=[2, 4, 1, 1, 1, 1])[0]
    if new_node_type == 1:
        expansion = ('Add', leaf_node, leaf_b)
    elif new_node_type == 2:
        expansion = ('Mul', leaf_node, leaf_b)
    elif new_node_type == 3:
        expansion = ('Pow', leaf_node, random.choice([-1, 1, 2]))
    elif new_node_type == 4:
        expansion = ('Exp', leaf_node)
    elif new_node_type == 5:
        expansion = ('Sin', leaf_node)
    elif new_node_type == 6:
        expansion = ('Cos', leaf_node)

    return replace_at_path(kernel, path, expansion)


def _symbolic_prove_operator(kernel_sym, tmpl, unknown_sym, dataset):
    """
    수치적으로 우수한 연산자에 대해 상위 개체 한정,
    SymPy 기호 적분(Symbolic Integration)을 수행하고
    CSV의 출력 함수(Target)와 기호/초정밀 일치하는지 타임아웃 2초 내에 증명.
    대수형/미분형 연산자도 올바르게 처리.
    """
    t_sym = sympy.Symbol('t')
    f_t_sym = sympy.Symbol('f_t')
    
    # 템플릿 유형 판별
    is_integral_tmpl = tmpl.has(sympy.Integral)
    
    for row in dataset:
        f_in_expr = row.get('input_expr')
        target_expr = row.get('target_expr')
        if f_in_expr is None or target_expr is None:
            return False, None

        # 1. 기호 연산자 형태 조립
        try:
            op_for_row = tmpl.subs(unknown_sym, kernel_sym).subs(f_t_sym, f_in_expr)
        except Exception:
            return False, None
            
        def _do_doit():
            return op_for_row.doit(conds='none')
            
        # 2. Timeout 2초 내 doit 시도
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_doit)
        try:
            evaluated_res = future.result(timeout=2.0)
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            return False, None
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            return False, None
        executor.shutdown(wait=False)
            
        # 적분형에서 적분이 실패하고 여전히 Integral이면 증명 실패
        if is_integral_tmpl and evaluated_res.has(sympy.Integral):
            return False, None
            
        # 3. 초정밀 수치 검증(1e-6 오차)
        symbols_in_res = list(evaluated_res.free_symbols)
        test_syms = list(set(symbols_in_res) | set(target_expr.free_symbols))
        # 적분형에서만 t를 제거 (적분 후 t가 소거된 경우)
        # 대수형/미분형에서는 t가 결과에 남아있으므로 제거하면 v1=0 오류 발생
        if is_integral_tmpl and t_sym in test_syms and t_sym not in evaluated_res.free_symbols:
            test_syms.remove(t_sym)
            
        for _ in range(5):
            pt_vals = [random.uniform(0.5, 3.0) for _ in test_syms]
            subs_dict = dict(zip(test_syms, pt_vals))
            try:
                v1_val = evaluated_res.subs(subs_dict).evalf()
                v2_val = target_expr.subs(subs_dict).evalf()
                v1 = complex(v1_val)
                v2 = complex(v2_val)
                
                if abs(v1 - v2) > 1e-6:
                    print(f"     [증명 실패] 값 불일치: v1={v1}, v2={v2}, diff={abs(v1 - v2)}")
                    return False, None
            except Exception as e:
                print(f"     [증명 에러] 수치 평가 불가 (특수 함수 잔존 등)")
                return False, None
                
    master_op = tmpl.subs(unknown_sym, kernel_sym)
    return True, master_op


def _run_ga_at_depth(
    depth: int,
    templates,
    terminal_pool: list,
    available_vars: list,
    injected,
    eliminated,
    local_dict,
    dataset,
    unknown_sym,
    pop_size: int,
    generations: int,
    eval_cache: dict,
    pruned_count_ref: list,   # [int] mutable reference
    seed_kernels: list,        # 이전 깊이에서 물려받은 우수 커널들
    logger,
    global_best: list,         # [score, operator] mutable reference
    is_complex_domain: bool = False,
    penalty_multiplier: float = 1.0,
    hints: dict = None,
    use_cache: bool = True,
) -> list:
    if hints is None: hints = {}
    """
    지정 depth에서 GA를 실행하고 상위 커널 리스트를 반환.
    seed_kernels가 있으면 빔 확장으로 초기 집단을 구성.
    """
    injected_names = [sym.name if hasattr(sym, 'name') else str(sym) for sym in injected]
    required_syms  = [s.name if hasattr(s, 'name') else str(s) for s in set(injected + eliminated)]

    print(f"\n\n{Colors.CYAN}{'━'*60}")
    print(f"{Colors.BOLD}  [Successive Deepening] 깊이 {depth} 탐색 시작{Colors.RESET}")
    if seed_kernels:
        print(f"     → 이전 깊이의 상위 커널 {len(seed_kernels)}개를 빔 확장으로 씨앗 사용")
    print(f"{Colors.CYAN}{'━'*60}{Colors.RESET}")

    if logger:
        logger.log_successive_deepening_start(depth, len(seed_kernels), global_best[0])

    population = []
    seen = set()

    # ── 씨앗이 있으면 직접 추가 또는 빔 확장으로 초기 집단 채우기 ──────────────
    if seed_kernels:
        # 먼저 seed kernel을 변이 없이 직접 population에 추가 (올바른 형태 보존)
        for base_kernel in seed_kernels:
            if len(population) >= pop_size:
                break
            for chosen_template in templates:
                if len(population) >= pop_size:
                    break
                expanded = simplify_tuple(base_kernel)
                # 변수 제약 처리
                for fs in list(get_free_symbols_from_tuple(expanded)):
                    if fs not in available_vars:
                        expanded = simplify_tuple(subs_in_tuple(expanded, fs, random.choice(available_vars)))
                missing = [v for v in injected_names if v not in get_free_symbols_from_tuple(expanded)]
                if not missing and passes_asymptotic_constraints(expanded, chosen_template):
                    key = str(expanded) + str(chosen_template)
                    if key not in seen:
                        seen.add(key)
                        population.append((chosen_template, expanded))

        # 그 다음 빔 확장으로 추가 개체 생성
        max_beam_tries = pop_size * 20
        for _ in range(max_beam_tries):
            if len(population) >= pop_size:
                break
            chosen_template = random.choice(templates)
            base_kernel = random.choice(seed_kernels)

            # 빔 확장 시도
            for __ in range(5):
                expanded = simplify_tuple(beam_expand_kernel(base_kernel, terminal_pool, available_vars))

                # 변수 제약 처리
                for fs in list(get_free_symbols_from_tuple(expanded)):
                    if fs not in available_vars:
                        expanded = simplify_tuple(subs_in_tuple(expanded, fs, random.choice(available_vars)))

                missing = [v for v in injected_names if v not in get_free_symbols_from_tuple(expanded)]
                if not missing and passes_asymptotic_constraints(expanded, chosen_template):
                    if 'I' in injected_names and not tuple_to_sympy(expanded, local_dict).has(sympy.I):
                        continue
                    key = str(expanded) + str(chosen_template)
                    if key not in seen:
                        seen.add(key)
                        population.append((chosen_template, expanded))
                        break

    # ── 부족한 자리는 랜덤 생성으로 채우기 ──────────────────────────────────────
    fill_attempts = 0
    while len(population) < pop_size and fill_attempts < pop_size * 50:
        fill_attempts += 1
        chosen_template = random.choice(templates)
        ind_tuple = simplify_tuple(random_tuple_operator(0, terminal_pool, depth))

        for fs in list(get_free_symbols_from_tuple(ind_tuple)):
            if fs not in available_vars:
                ind_tuple = simplify_tuple(subs_in_tuple(ind_tuple, fs, random.choice(available_vars)))

        missing = [v for v in injected_names if v not in get_free_symbols_from_tuple(ind_tuple)]
        if not missing and passes_asymptotic_constraints(ind_tuple, chosen_template):
            if 'I' in injected_names and not tuple_to_sympy(ind_tuple, local_dict).has(sympy.I):
                continue
            key = str(ind_tuple) + str(chosen_template)
            if key not in seen:
                seen.add(key)
                population.append((chosen_template, ind_tuple))

    # ── GA 루프 ─────────────────────────────────────────────────────────────────
    local_best_fit = -1.0
    local_stagnation = 0
    max_local_stagnation = max(3, generations) * pop_size * 3

    for gen in range(generations):
        gen_start_time = time.time()
        cache_hits = 0
        fitness_scores = []
        
        for idx, (tmpl, ind_tuple) in enumerate(population):
            if idx > 0 and idx % 100 == 0:
                print(f"[진행률] 깊이 {depth} | 세대 {gen} | {pruned_count_ref[0] + idx}개 평가 중...")
                # 일시정지 체크
                pause_file = os.environ.get("PAUSE_FILE")
                if pause_file:
                    while os.path.exists(pause_file):
                        time.sleep(1)
            
            if (str(ind_tuple) + str(tmpl)) in eval_cache:
                cache_hits += 1

            best_fit = -1.0
            best_tmpl = tmpl
            best_full_op = None

            for test_tmpl in templates:
                expr_str = str(ind_tuple) + str(test_tmpl)
                if expr_str in eval_cache and use_cache:
                    fit = eval_cache[expr_str]
                    if fit > best_fit:
                        best_fit = fit
                        best_tmpl = test_tmpl
                        try:
                            best_full_op = test_tmpl.subs(unknown_sym, tuple_to_sympy(ind_tuple, local_dict))
                        except Exception:
                            best_full_op = None
                    continue

                if not passes_asymptotic_constraints(ind_tuple, test_tmpl):
                    fit = 0.0
                    f_op = None
                else:
                    ind_sym = tuple_to_sympy(ind_tuple, local_dict)
                    if 'I' in injected_names and not ind_sym.has(sympy.I):
                        fit = 0.0
                        f_op = None
                    else:
                        try:
                            f_op = test_tmpl.subs(unknown_sym, ind_sym)
                            # [Bug 3 수정] isinstance는 최상위 노드만 보지만 has()는 트리 전체를 탐색
                            # Unknown*Derivative(f_t,t) 같은 Mul 래핑 케이스도 올바르게 판별
                            is_algebraic = not f_op.has(Integral) and not f_op.has(Derivative)
                            if is_algebraic and 'f_t' not in get_free_symbols_from_tuple(ind_tuple):
                                fit = 0.0
                            else:
                                raw_fit = dispatch_fitness(f_op, dataset, is_complex_domain)
                                missing_count = sum(1 for rs in required_syms if rs not in get_free_symbols_from_tuple(ind_tuple))
                                missing_domain_penalty = 10.0 * missing_count
                                
                                # [신규] 힌트 필수 심볼 체크
                                forced_penalty = 0.0
                                if hints.get("trig"):
                                    t_type = hints["trig"].get("type")
                                    if t_type == "cos" and not f_op.has(sympy.cos): forced_penalty += 50.0
                                    elif t_type == "sin" and not f_op.has(sympy.sin): forced_penalty += 50.0
                                    elif t_type == "tan" and not f_op.has(sympy.tan): forced_penalty += 50.0
                                
                                if hints.get("exp"):
                                    e_type = hints["exp"].get("type")
                                    if e_type == "exp" and not f_op.has(sympy.exp): forced_penalty += 50.0
                                    elif e_type == "pow" and not f_op.has(sympy.Pow): forced_penalty += 50.0
                                    
                                if hints.get("log"):
                                    if not f_op.has(sympy.log): forced_penalty += 50.0
                                    
                                complexity = len(str(ind_tuple))
                                base_penalty = 0.003 if is_algebraic else 0.0001
                                penalty_weight = base_penalty * penalty_multiplier
                                fit = max(0.0, raw_fit - missing_domain_penalty - forced_penalty - penalty_weight * complexity)
                        except Exception:
                            fit = 0.0
                            f_op = None

                if use_cache:
                    eval_cache[expr_str] = fit
                if fit > best_fit:
                    best_fit = fit
                    best_tmpl = test_tmpl
                    best_full_op = f_op

            if best_fit < 0.0:
                best_fit = 0.0

            fitness_scores.append((best_fit, best_tmpl, ind_tuple, best_full_op))

        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        # ── [하이브리드 2차 기호 증명: 상위 5개 엘리트 검증] ──────────────────────
        if fitness_scores[0][0] > 0.0:
            for elite_idx in range(min(5, len(fitness_scores))):
                elite_score, tmpl, elite_kernel, full_op = fitness_scores[elite_idx]
                if elite_score > 0.0:
                    ind_sym = tuple_to_sympy(elite_kernel, local_dict)
                    is_sym_perfect, proven_op = _symbolic_prove_operator(ind_sym, tmpl, unknown_sym, dataset)
                    if is_sym_perfect:
                        print(f"\n  {Colors.GREEN}{Colors.BOLD}[진리 증명 성공!] 대수적 적분이 완벽하게 일치합니다!{Colors.RESET}")
                        print(f"     => 궁극의 도출 공식: {Colors.YELLOW}{proven_op}{Colors.RESET}")
                        fitness_scores[elite_idx] = (100.0, tmpl, elite_kernel, proven_op)
                        fitness_scores.sort(key=lambda x: x[0], reverse=True)
                        break

        pruned_count_ref[0] += len(population)
        local_stagnation += len(population)

        pop_best_fit   = fitness_scores[0][0]
        pop_best_kernel = fitness_scores[0][2]
        pop_best_full   = fitness_scores[0][3]

        gen_time = time.time() - gen_start_time
        hit_rate = (cache_hits / pop_size) * 100.0

        if logger:
            logger.log_generation_summary(depth, gen, pruned_count_ref[0], hit_rate, pop_best_fit)
            
        print(f"  - {Colors.BLUE}[깊이 {depth} | 세대 {gen} 완료] {Colors.RESET}누적 방출: {pruned_count_ref[0]}개 | 최고점: {pop_best_fit:.4f} | 소요시간: {gen_time:.2f}초 | 캐시 재사용: {hit_rate:.1f}%")

        # 전역 최고 갱신
        if pop_best_fit > global_best[0]:
            global_best[0] = pop_best_fit
            global_best[1] = pop_best_full
            global_best[2] = pop_best_kernel
            local_stagnation = 0
            print(f"\n  {Colors.MAGENTA}{Colors.BOLD}[전역 최고해 갱신 (Global Best)! 깊이 {depth} | 세대 {gen}] {Colors.RESET}"
                  f"Score: {global_best[0]:.4f}  Kernel: {pop_best_kernel}")
            if logger:
                logger.log_generation(depth, gen, global_best[0], global_best[1])
            if global_best[0] >= 99.0:
                break

        # 깊이 내 지역 최고 갱신
        if pop_best_fit > local_best_fit:
            local_best_fit = pop_best_fit
            local_stagnation = 0

        if local_stagnation > max_local_stagnation:
            print(f"\n  {Colors.RED}[Depth {depth}] 지역 정체 감지 ({local_stagnation}회). 다음 심층부로 전환합니다.{Colors.RESET}")
            break

        # ── 다음 세대 구성 ────────────────────────────────────────────────────
        top_k = max(1, int(len(fitness_scores) * 0.2))
        next_population = [(fs[1], fs[2]) for fs in fitness_scores[:top_k]]

        while len(next_population) < pop_size:
            chosen_template = random.choice(templates)
            child_kernel = None

            for _ in range(10):
                roll = random.random()
                if roll < 0.5 and len(fitness_scores) >= 2:
                    # 교차
                    p1 = random.choice(fitness_scores[:top_k])[2]
                    p2 = random.choice(fitness_scores[:top_k])[2]
                    test_ind = simplify_tuple(tuple_homologous_crossover(p1, p2))
                elif roll < 0.7 and seed_kernels:
                    # 빔 확장 (우수 씨앗 재활용)
                    base = random.choice(seed_kernels)
                    test_ind = simplify_tuple(beam_expand_kernel(base, terminal_pool, available_vars))
                else:
                    # 랜덤 생성
                    while True:
                        test_ind = simplify_tuple(random_tuple_operator(0, terminal_pool, depth))
                        for fs in list(get_free_symbols_from_tuple(test_ind)):
                            if fs not in available_vars:
                                test_ind = simplify_tuple(subs_in_tuple(test_ind, fs, random.choice(available_vars)))
                        if not [v for v in injected_names if v not in get_free_symbols_from_tuple(test_ind)]:
                            if passes_asymptotic_constraints(test_ind, chosen_template):
                                break

                for fs in list(get_free_symbols_from_tuple(test_ind)):
                    if fs not in available_vars:
                        test_ind = simplify_tuple(subs_in_tuple(test_ind, fs, random.choice(available_vars)))

                test_str = str(test_ind) + str(chosen_template)
                if test_str not in eval_cache:
                    kernel_expr = tuple_to_sympy(test_ind, local_dict)
                    if prevents_inverse_cancellation(kernel_expr, chosen_template, unknown_sym):
                        child_kernel = test_ind
                        break

            if child_kernel is None:
                child_kernel = test_ind

            next_population.append((chosen_template, child_kernel))

        population = next_population
    # 상위 커널 반환: 종 분화(Speciation) 적용하여 다양성 보존
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    
    top_survivors = []
    seen_species = set()
    seed_quota = max(15, int(len(fitness_scores) * 0.1))  # 최소 15개, 기본 10%
    
    for fs in fitness_scores:
        tup_kernel = fs[2]
        # fs[2] is child_kernel tuple
        species_id = get_species_id(tup_kernel, available_vars)
        
        if species_id == "INVALID":
            continue
            
        if species_id not in seen_species:
            seen_species.add(species_id)
            top_survivors.append(tup_kernel)
            
        # 쿼터 다 차면 종료
        if len(top_survivors) >= seed_quota:
            break
            
    # 만약 유효한 종이 너무 적어서 쿼터를 못 채우면 나머지는 점수 순으로 그냥 채움
    if len(top_survivors) < seed_quota:
        for fs in fitness_scores:
            tup_kernel = fs[2]
            if tup_kernel not in top_survivors:
                top_survivors.append(tup_kernel)
            if len(top_survivors) >= seed_quota:
                break
                
    return top_survivors


# ── 대수형 쇼트컷: 유리식 치환으로 연산자 직접 추출 ─────────────────────────────
def _try_single_row_extraction(input_expr, output_expr):
    """단일 행에서 ph 플레이스홀더로 연산자를 추출. 실패시 None."""
    ph = sympy.Symbol('__ph__')

    def _is_clean(expr):
        if not isinstance(expr, sympy.Expr) or not expr.has(ph):
            return False
        return True

    # 치환 쌍 목록 구성 (직접 + 역수 + 거듭제곱 + 지수 켤레)
    subs_list = [(input_expr, ph)]
    try:
        recip = sympy.cancel(1 / input_expr)
        if recip != input_expr:
            subs_list.append((recip, 1 / ph))
    except Exception:
        pass
    for n in [2, 3, -2, -3]:
        try:
            subs_list.append((sympy.cancel(input_expr ** n), ph ** n))
        except Exception:
            pass
    # 지수 켤레: input에 exp(-X)가 있으면 output의 exp(+X)를 coeff/ph 로 치환
    # 예) input = a*exp(-k*t) → exp(k*t) = a/ph
    try:
        for atom in input_expr.atoms(sympy.exp):
            conj = sympy.exp(-atom.args[0])
            coeff = sympy.cancel(input_expr / atom)
            subs_list.append((conj, sympy.cancel(coeff / ph)))
    except Exception:
        pass

    for use_cancel_first in [False, True]:
        target = sympy.cancel(output_expr) if use_cancel_first else output_expr
        try:
            res = sympy.cancel(target.subs(subs_list))
            if _is_clean(res):
                return res
        except Exception:
            pass
    return None


def try_rational_substitution(dataset):
    """
    모든 행에서 동일한 유리식 연산자를 직접 추출.
    성공시 f_t로 표현된 연산자 반환, 실패시 None → GA 폴백.
    """
    ph = sympy.Symbol('__ph__')
    f_t_sym = sympy.Symbol('f_t')
    extracted = []

    for row in dataset:
        in_e = row.get('input_expr')
        out_e = row.get('target_expr')
        if in_e is None or out_e is None:
            return None
        op = _try_single_row_extraction(in_e, out_e)
        if op is None:
            return None
        extracted.append(op)

    if not extracted:
        return None

    # 일관성 수치 검증 (5회)
    ref_op = extracted[0]
    all_params = list(ref_op.free_symbols - {ph})
    valid_checks = 0

    for _ in range(5):
        ph_val = random.uniform(0.3, 2.0)
        pvals = {p: random.uniform(0.5, 2.0) for p in all_params}
        subs_d = {ph: ph_val, **pvals}
        try:
            ref_v = complex(ref_op.subs(subs_d).evalf())
            if not (abs(ref_v) < 1e10):
                continue
        except Exception:
            continue
            
        for other in extracted[1:]:
            try:
                od = {ph: ph_val, **{p: pvals.get(p, random.uniform(0.5, 2.0)) for p in other.free_symbols - {ph}}}
                ov = complex(other.subs(od).evalf())
                if abs(ref_v - ov) > 1e-6:
                    return None
            except Exception:
                return None
                
        valid_checks += 1

    if valid_checks == 0:
        return None

    return ref_op.subs(ph, f_t_sym)


def try_derivative_shortcut(dataset):
    """
    모든 행에서 output == d^n/dvar^n(input)이 일관되게 성립하는 (var, n) 쌍을 탐색.
    대수형 쇼트컷과 마찬가지로 GA를 완전히 우회하며, 성공 시 Derivative(f_t, var, n) 반환.
    1계(n=1) → 2계(n=2) 순으로 탐색하며, 모든 행이 일치하지 않으면 None 반환 (GA 폴백).
    """
    f_t_sym = sympy.Symbol('f_t')
    ignore = {f_t_sym, sympy.pi, sympy.E, sympy.I}

    # 후보 변수: 모든 행 입력 표현식의 자유 변수 합집합
    candidate_vars = set()
    for row in dataset:
        in_expr = row.get('input_expr')
        if in_expr is not None:
            candidate_vars.update(in_expr.free_symbols - ignore)

    if not candidate_vars:
        return None

    for order in [1, 2]:
        for var in sorted(candidate_vars, key=lambda s: s.name):
            all_rows_match = True
            for row in dataset:
                in_expr = row.get('input_expr')
                out_expr = row.get('target_expr')
                if in_expr is None or out_expr is None:
                    all_rows_match = False
                    break
                try:
                    computed = sympy.diff(in_expr, var, order)
                    # 1차: 기호 일치 확인 (expand로 정규화)
                    if sympy.expand(computed - out_expr) == 0:
                        continue  # 이 행은 일치 ✅
                    # 2차: simplify 시도 (약간 더 무거움)
                    if sympy.simplify(computed - out_expr) == 0:
                        continue  # 이 행은 일치 ✅
                    # 3차: 수치 검증 (기호 단순화가 실패한 경우)
                    all_free = list((in_expr.free_symbols | out_expr.free_symbols) - ignore)
                    numeric_match = True
                    for _ in range(5):
                        pt = {s: random.uniform(0.3, 2.0) for s in all_free}
                        v1 = complex(computed.subs(pt).evalf())
                        v2 = complex(out_expr.subs(pt).evalf())
                        if abs(v1 - v2) > 1e-6:
                            numeric_match = False
                            break
                    if not numeric_match:
                        all_rows_match = False
                        break
                except Exception:
                    all_rows_match = False
                    break

            if all_rows_match:
                # order=1 이면 Derivative(f_t, var), order=2 이면 Derivative(f_t, (var, 2))
                if order == 1:
                    return sympy.Derivative(f_t_sym, var)
                else:
                    return sympy.Derivative(f_t_sym, (var, order))

    return None


def generate_symbolic_operator(dataset_df, pop_size=30, generations=10, penalty_multiplier=1.0, hints=None, use_shortcuts=True, use_cache=True, use_pruning=True) -> sympy.Expr:
    if hints is None: hints = {}

    start_time = time.time()
    raw_dataset = dataset_df.to_dict('records')
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    transformations = (standard_transformations + (implicit_multiplication_application,))
    
    local_dict = {
        't': t, 's': s, 'a': a, 'b': b, 
        'f': freq, 'freq': freq, 'w': omega, 'omega': omega, 'theta': theta, 
        'T': period, 'period': period, 'k': k_wave, 'zeta': zeta, 'tau': tau,
        'c': c_light, 'g': g_grav, 'G': G_grav, 'k_B': k_B, 
        'hbar': h_bar, 'Z_0': Z_0, 'mu_0': mu_0, 'eps_0': eps_0,
        'pi': sympy.pi, 'e': sympy.E, 'I': sympy.I, 'j': sympy.I
    }
    
    dataset = []
    for row in raw_dataset:
        try:
            in_expr = parse_expr(str(row['input']), local_dict=local_dict, transformations=transformations)
            out_expr = parse_expr(str(row['output']), local_dict=local_dict, transformations=transformations)
            roc_expr = None
            if 'roc' in row and isinstance(row['roc'], str) and row['roc'].strip():
                roc_expr = parse_expr(str(row['roc']), local_dict=local_dict, transformations=transformations)
            dataset.append({'input_expr': in_expr, 'target_expr': out_expr, 'roc': roc_expr})
        except:
            pass

    # ── [신규] domain 열 파싱: 적분 구간 직접 지정 ────────────────────────────────
    domain_bounds = None
    if raw_dataset and 'domain' in raw_dataset[0]:
        domain_str = str(raw_dataset[0].get('domain', '')).strip()
        if domain_str:
            try:
                parts = domain_str.split(',')
                if len(parts) == 2:
                    lower_expr = parse_expr(parts[0].strip(), local_dict=local_dict, transformations=transformations)
                    upper_expr = parse_expr(parts[1].strip(), local_dict=local_dict, transformations=transformations)
                    domain_bounds = (lower_expr, upper_expr)
                    print(f"  {Colors.CYAN}[domain 열 감지] 적분 구간: [{lower_expr}, {upper_expr}]{Colors.RESET}")
            except Exception:
                print(f"  {Colors.YELLOW}[domain 열] 파싱 실패 — 기본 구간 사용{Colors.RESET}")

    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "evolution_trace.md")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = GlassBoxLogger(log_path)

    # ── [Symbolic Shortcut 우선 적용] 공리 연역 전 빠른 경로 탐색 ─────────────────
    # 적분 변수 존재 여부를 판단하기 위한 경량 변수 분류 (물리 상수 제외)
    _ignore_quick = {sympy.E, sympy.pi, sympy.I}
    _quick_in  = set()
    _quick_out = set()
    for _row in dataset:
        if _row.get('input_expr'):
            _quick_in.update(_row['input_expr'].free_symbols - _ignore_quick)
        if _row.get('target_expr'):
            _quick_out.update(_row['target_expr'].free_symbols - _ignore_quick)
    _quick_eliminated = _quick_in - _quick_out  # 입력에만 있는 변수 (적분 변수 후보)

    if use_shortcuts and not _quick_eliminated:
        # 미분형 쇼트컷 우선 시도 (d/dt 검출)
        print(f"\n  {Colors.CYAN}[심볼릭 쇼트컷 1/2] 미분 연산자 직접 검출 시도 중...{Colors.RESET}")
        shortcut_deriv = try_derivative_shortcut(dataset)
        if shortcut_deriv is not None:
            elapsed = time.time() - start_time
            print(f"\n  {Colors.GREEN}[미분 쇼트컷 성공] 미분 연산자 직접 추출 완료 ({elapsed:.2f}초): {shortcut_deriv}{Colors.RESET}")
            logger.log_completion(elapsed, shortcut_deriv, 100.0, 0)
            logger.log_shortcut_result(shortcut_deriv, dataset, elapsed)
            return shortcut_deriv
        else:
            print(f"  {Colors.YELLOW}→ 미분 쇼트컷 실패. 대수형 쇼트컷 시도합니다.{Colors.RESET}")

        # 대수형 쇼트컷 시도 (유리식 직접 치환)
        print(f"  {Colors.CYAN}[심볼릭 쇼트컷 2/2] 유리식 직접 치환 추출 시도 중...{Colors.RESET}")
        shortcut_op = try_rational_substitution(dataset)
        if shortcut_op is not None:
            elapsed = time.time() - start_time
            print(f"\n  {Colors.GREEN}[대수 쇼트컷 성공] 대수 연산자 직접 추출 완료 ({elapsed:.2f}초): {shortcut_op}{Colors.RESET}")
            logger.log_completion(elapsed, shortcut_op, 100.0, 0)
            logger.log_shortcut_result(shortcut_op, dataset, elapsed)
            return shortcut_op
        else:
            print(f"  {Colors.YELLOW}→ 쇼트컷 전체 실패. GA 탐색으로 전환합니다.{Colors.RESET}")
    elif not use_shortcuts:
        print(f"  {Colors.YELLOW}[심볼릭 쇼트컷 비활성화] GA 탐색으로 직접 진입합니다.{Colors.RESET}")

    # ── 공리 연역 (쇼트컷 실패 또는 적분형 데이터) ──────────────────────────────────
    if use_pruning:
        deductive_engine = AxiomaticDeductiveEngine()
        templates, unknown_sym, injected, eliminated, raw_symbols, is_complex_domain = deductive_engine.analyze_dataset(dataset, domain_bounds=domain_bounds)
        logger.log_deduction(templates)
    else:
        # 공리 가지치기 비활성화: 구조적 템플릿 없이 범용 Unknown 하나로만 탐색
        unknown_sym = sympy.Symbol('Unknown')
        _constants  = {sympy.E, sympy.pi, sympy.I}
        _all_in, _all_out = set(), set()
        for _row in dataset:
            if _row.get('input_expr'):  _all_in.update(_row['input_expr'].free_symbols  - _constants)
            if _row.get('target_expr'): _all_out.update(_row['target_expr'].free_symbols - _constants)
        eliminated       = list(_all_in  - _all_out)
        injected         = list(_all_out - _all_in)
        raw_symbols      = list(_all_in  | _all_out)
        is_complex_domain = False
        templates        = [unknown_sym]   # 구조 제약 없는 범용 템플릿 1개
        print(f"  {Colors.YELLOW}[공리 가지치기 비활성화] 범용 템플릿 1개로 탐색 (구조적 제약 없음){Colors.RESET}")

    base_numbers = [1, 0, -1, 2]
    
    # [신규] 힌트 상수를 풀에 쑤셔넣기
    if hints:
        for func_name, params in hints.items():
            if params:
                for key, val_str in params.items():
                    if key not in ["type", "freq_type"] and val_str and val_str != "0":
                        try:
                            # 특수 처리: 일반 주파수(f)인 경우 2*pi*f 형태의 상수를 풀에 우선 추가
                            if key == "freq" and params.get("freq_type") == "freq":
                                val_sym = sympy.sympify(val_str)
                                f_val = 2 * sympy.pi * val_sym
                                if str(f_val) not in raw_symbols:
                                    raw_symbols.append(f_val)
                            
                            val_sym = sympy.sympify(val_str)
                            if val_sym.is_number:
                                num = float(val_sym) if '.' in str(val_sym) else int(val_sym)
                                if num not in base_numbers:
                                    base_numbers.append(num)
                            else:
                                if str(val_sym) not in raw_symbols:
                                    raw_symbols.append(val_sym)
                        except Exception:
                            pass

    terminal_pool = list(set(base_numbers + [sym.name if hasattr(sym, 'name') else str(sym) for sym in raw_symbols] + ['a', 'b']))

    import math
    required_var_count = len(set(injected + eliminated))
    computed_min_depth = math.ceil(math.log2(max(required_var_count, 2)))
    print(f"  {Colors.CYAN}[유전자 상자 초기화] {terminal_pool}")
    print(f"  [차원 체인] 필수 도메인 변수 {required_var_count}개 → 최소 시작 깊이: {computed_min_depth}{Colors.RESET}")
    var_name_list = [sym.name if hasattr(sym, 'name') else str(sym) for sym in set(injected + eliminated)]
    logger.log_terminal_pool(terminal_pool)

    # available_vars: GA kernel can freely use these variable names.
    # Must include raw_symbols (retained vars like a, t, omega, k) in addition to injected+eliminated.
    # Without this, GA kernels containing retained vars get forcibly replaced.
    injected_elim_names = set(
        sym.name if hasattr(sym, 'name') else str(sym)
        for sym in list(set(injected + eliminated))
    )
    raw_sym_names = set(
        sym.name if hasattr(sym, 'name') else str(sym)
        for sym in raw_symbols
        if isinstance(sym, sympy.Symbol)
    )
    available_vars = list(injected_elim_names | raw_sym_names)

    # f_t must be in terminal_pool and available_vars for algebraic/derivative searches.
    if 'f_t' not in terminal_pool:
        terminal_pool.append('f_t')

    has_algebraic_tmpl = any(
        not tmpl.has(Integral) and not tmpl.has(Derivative)
        for tmpl in templates
    )
    if has_algebraic_tmpl or not eliminated:
        if 'f_t' not in available_vars:
            available_vars.append('f_t')

    if not available_vars:
        available_vars = ['t', 's']

    # ── Successive Deepening 파라미터 ────────────────────────────────────────────
    max_depth = computed_min_depth + 6   # 최대 깊이 제한 완전 해제 (충분히 깊게 탐색)
    gens_per_depth = max(3, generations // (max_depth - computed_min_depth + 1))
    
    logger.log_dimension_chain(required_var_count, computed_min_depth, var_name_list, pop_size, gens_per_depth)

    eval_cache = {}
    pruned_count_ref = [0]          # mutable int reference
    global_best = [-1.0, None, None] # [score, operator, tuple_kernel]
    seed_kernels = []               # 첫 번째 깊이는 씨앗 없이 시작

    # -- Algebraic seed: injected vars (e.g. a, b) + f_t 포함 seed 추가 --
    # algebraic template이 있고 injected vars가 있을 때, Lorentzian 형태 seed를
    # 초기에 제공하여 GA 탐색 효율을 높인다.
    has_algebraic_tmpl = any(
        not tmpl.has(Integral) and not tmpl.has(Derivative)
        for tmpl in templates
    )
    injected_names_for_seed = [sym.name if hasattr(sym, 'name') else str(sym) for sym in injected]
    if has_algebraic_tmpl and len(injected_names_for_seed) >= 1 and 'f_t' in available_vars:
        # Generate diverse seed kernels covering algebraic Lorentzian-like structures
        _a = injected_names_for_seed[0] if len(injected_names_for_seed) > 0 else 'a'
        _b = injected_names_for_seed[1] if len(injected_names_for_seed) > 1 else 'b'
        _seeds = [
            # a / (b^2 + f_t^2)  -- classic Lorentzian
            ('Mul', _a, ('Pow', ('Add', ('Pow', _b, 2), ('Pow', 'f_t', 2)), -1)),
            # a / (b + f_t^2)
            ('Mul', _a, ('Pow', ('Add', _b, ('Pow', 'f_t', 2)), -1)),
            # a / (b^2 + f_t)
            ('Mul', _a, ('Pow', ('Add', ('Pow', _b, 2), 'f_t'), -1)),
            # a * f_t / (b^2 + f_t^2)  -- full Lorentzian (for add template)
            ('Mul', _a, 'f_t', ('Pow', ('Add', ('Pow', _b, 2), ('Pow', 'f_t', 2)), -1)),
            # a / (b + f_t)
            ('Mul', _a, ('Pow', ('Add', _b, 'f_t'), -1)),
        ]
        seed_kernels.extend(_seeds)
        print(f'  [Algebraic Seed] {len(_seeds)}개 Lorentzian 형태 seed 커널 주입')

    print(f"\n  {Colors.CYAN}[Successive Deepening 전략] 깊이 {computed_min_depth} → {max_depth}  "
          f"(깊이당 최대 {gens_per_depth}세대){Colors.RESET}")

    for depth in range(computed_min_depth, max_depth + 1):
        survivors = _run_ga_at_depth(
            depth=depth,
            templates=templates,
            terminal_pool=terminal_pool,
            available_vars=available_vars,
            injected=injected,
            eliminated=eliminated,
            local_dict=local_dict,
            dataset=dataset,
            unknown_sym=unknown_sym,
            pop_size=pop_size,
            generations=gens_per_depth,
            eval_cache=eval_cache,
            pruned_count_ref=pruned_count_ref,
            seed_kernels=seed_kernels,
            logger=logger,
            global_best=global_best,
            is_complex_domain=is_complex_domain,
            penalty_multiplier=penalty_multiplier,
            hints=hints,
            use_cache=use_cache,
        )
        seed_kernels = survivors

        if global_best[0] >= 99.0:
            print(f"\n  {Colors.GREEN}{Colors.BOLD}[완벽한 해 발견!] Successive Deepening 조기 종료.{Colors.RESET}")
            break

    def get_node_count(tup):
        if not isinstance(tup, tuple): return 1
        return 1 + sum(get_node_count(c) for c in tup[1:])
        
    node_count = get_node_count(global_best[2]) if global_best[2] is not None else 0
    elapsed_final = time.time() - start_time
    logger.log_completion(elapsed_final, global_best[1], global_best[0], node_count)
    logger.log_ga_result(global_best[1], elapsed_final, pruned_count_ref[0], global_best[0])
    return global_best[1]

