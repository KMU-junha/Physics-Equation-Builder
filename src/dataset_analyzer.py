import pandas as pd
from typing import Literal

def detect_dataset_type(df: pd.DataFrame) -> Literal["NUMERIC", "SYMBOLIC"]:
    """
    CSV 데이터 프레임이 순수 숫자 값(운동학 데이터 등)인지, 
    수식/알파벳 등 기호 변환 테이블(라플라스 등)인지 감지합니다.
    """
    # 데이터 프레임의 모든 칼럼을 훑어보면서 문자열 연산자나 변수명(알파벳)이 있는지 검사
    for col in df.columns:
        # object/string 타입으로 인식된 경우 내부를 들여다봅니다
        if df[col].dtype == object or df[col].dtype.name == 'string':
            for val in df[col].dropna():
                val_str = str(val).strip()
                # 숫자로 변환이 안되거나 알파벳이 섞인 문자열이 하나라도 있으면 심볼릭으로 간주
                try:
                    float(val_str)
                except ValueError:
                    return "SYMBOLIC"
    
    # 예외가 전혀 안 발생했다면 전부 변환 가능한 Numeric 데이터임
    return "NUMERIC"
