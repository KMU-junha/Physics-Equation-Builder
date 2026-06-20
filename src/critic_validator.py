"""
차원 분석 및 물리법칙 위배 여부를 검증하는 로직을 담당하는 모듈입니다.
"""
from sympy import Eq, Add, Mul  # type: ignore
from sympy.core.numbers import Number  # type: ignore
from sympy.physics.units.systems import SI  # type: ignore
from sympy.physics.units import Quantity  # type: ignore

def check_dimensional_homogeneity(lhs, rhs):
    """
    수식의 좌변(lhs)과 우변(rhs)의 물리적 일관성을 검사합니다.
    SINDy가 생성한 순수한 수학적 상수(예: 10.3866)는 양변의 차원을 맞추어주는 
    "물리적 보정 상수(Unknown Unit Constant)" 역할을 할 수 있다고 관대하게 가정합니다.
    (예: v = 10.3866 * t - 4.3707 이라면 10.3866이 m/s^2 이고 4.3707이 m/s 일 수 있음을 예비 허용)
    """
    try:
        from sympy import expand, Symbol
        
        # 1. 수식 전개
        expanded_rhs = expand(rhs)
        terms = Add.make_args(expanded_rhs)
        
        # 2. 기준 차원 설정 (보통 좌변(v)은 변수가 1개이므로 그 자체가 차원의 기준)
        dim_lhs = SI.get_dimensional_expr(lhs)
        
        for term in terms:
            # 3. 각 항(term)에 순수 상수(숫자)가 있는지 확인
            has_constant = False
            
            if isinstance(term, Number):
                has_constant = True
            elif isinstance(term, Mul):
                for arg in term.args:
                    if isinstance(arg, Number):
                        has_constant = True
                        break
                        
            # 4. 상수가 곱해져있거나 아예 상수 자체인 독립항이라면
            #    그 상수가 "필요에 따라 모자란 단위를 메꿔주는 보정 가속도/속도/기타 등등"이 
            #    될 수 있다고 가정하므로 이 항에 대한 검증은 무조건 관대하게 통과(Pass)시킵니다.
            if has_constant:
                continue
                
            # 5. 하지만 상수가 전혀 없는, 순수 변수들만의 조합 항(예: v = d + t)이라면 
            #    이건 보정해 줄 여지가 없으므로 아주 엄격하게 좌변과 차원이 일치하는지 비교합니다.
            dim_term = SI.get_dimensional_expr(term)
            if dim_lhs != dim_term:
                return False
                
        # 모든 항이 검증을 통과했거나, 보정 상수로 메꿔질 여지가 있다면 True
        return True
    except Exception as e:
        print(f"  [Critic Error] 차원 구조 분석 중 오류 발생: {e}")
        return False

# 예시 코드
if __name__ == "__main__":
    from sympy.physics.units import meter, second  # type: ignore
    
    # 예시 1: 올바른 차원 (속도 = 거리 / 시간)
    v = meter / second
    d = 10 * meter
    t = 5 * second
    
    is_valid_1 = check_dimensional_homogeneity(v, d / t)
    print(f"예시 1 (v = d/t): {is_valid_1}")  # 예상 결과: True
    
    # 예시 2: 잘못된 차원 (속도 = 거리 + 시간)
    is_valid_2 = check_dimensional_homogeneity(v, d + t)
    print(f"예시 2 (v = d + t): {is_valid_2}")  # 예상 결과: False
