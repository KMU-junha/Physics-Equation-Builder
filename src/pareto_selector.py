"""
정확도 vs 물리법칙 타협점을 찾아 파레토 프론트를 구성하는 로직을 담당하는 모듈입니다.
실제 MSE를 활용하여 수식의 품질을 평가합니다.
"""

from typing import Dict, List, Tuple
import numpy as np
from sympy import count_ops, Eq  # pyre-ignore[21]


def calculate_complexity(equation: Eq) -> int:
    """
    주어진 sympy 수식(Eq)의 복잡도를 계산합니다.
    수식에 포함된 연산자 개수(Add, Mul, Pow 등)를 반환합니다.

    Args:
        equation (sympy.Eq): 복잡도를 계산할 대상 수식

    Returns:
        int: 수식의 복잡도(연산자 개수) 점수
    """
    return count_ops(equation)


def select_best_equations(
    equations: List[Eq],
    mse_values: List[float],
) -> Dict[Eq, Tuple[int, float]]:
    """
    차원 검증을 통과한 수식 리스트와 각각의 실제 MSE를 받아,
    복잡도와 MSE 점수를 묶어서 반환합니다.

    Args:
        equations (List[sympy.Eq]): 평가할 수식 리스트
        mse_values (List[float]): 각 수식에 대응하는 실제 MSE 값 리스트

    Returns:
        Dict[sympy.Eq, Tuple[int, float]]:
            수식을 키(key), (복잡도, MSE) 튜플을 값(value)으로 하는 딕셔너리.
            MSE 오름차순으로 정렬됩니다.
    """
    if len(equations) != len(mse_values):
        raise ValueError(
            f"equations({len(equations)})와 mse_values({len(mse_values)})의 길이가 다릅니다."
        )

    evaluation_results = {}
    for eq, mse in zip(equations, mse_values):
        complexity_score = calculate_complexity(eq)
        evaluation_results[eq] = (complexity_score, float(mse))

    # MSE 오름차순 정렬 (낮을수록 좋음)
    sorted_results = dict(
        sorted(evaluation_results.items(), key=lambda item: item[1][1])
    )
    return sorted_results
