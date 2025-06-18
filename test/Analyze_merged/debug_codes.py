# 기준_년분기_코드 형식 확인용 디버그 코드

import os
import pandas as pd

# 1. LocalPeople 샘플 확인
data_dir = os.path.abspath("../../Data/LocalPeople")
sample_file = os.path.join(data_dir, "LOCAL_PEOPLE_DONG_201901.csv")

print("=== LocalPeople 파일 읽기 테스트 ===")

# 1. 현재 merged.py 방식
print("\n1. 현재 merged.py 방식:")
try:
    df1 = pd.read_csv(sample_file, encoding="utf-8", dtype={"행정동코드": str})
    print(f"   행 수: {len(df1)}")
    print(f"   컬럼 수: {len(df1.columns)}")
    print(f"   첫 5개 컬럼: {df1.columns[:5].tolist()}")
    print(f"   행정동코드 샘플: {df1['행정동코드'].head(3).tolist()}")
    print(f"   총생활인구수 샘플: {df1['총생활인구수'].head(3).tolist()}")
except Exception as e:
    print(f"   에러: {e}")

# 2. Analyze_LocalPeople.py 방식 모방
print("\n2. Analyze_LocalPeople.py 방식:")
try:
    expected_cols = [
        "기준일ID",
        "시간대구분",
        "행정동코드",
        "총생활인구수",
        "남자0세부터9세생활인구수",
        "남자10세부터14세생활인구수",
        "남자15세부터19세생활인구수",
        "남자20세부터24세생활인구수",
        "남자25세부터29세생활인구수",
        "남자30세부터34세생활인구수",
        "남자35세부터39세생활인구수",
        "남자40세부터44세생활인구수",
        "남자45세부터49세생활인구수",
        "남자50세부터54세생활인구수",
        "남자55세부터59세생활인구수",
        "남자60세부터64세생활인구수",
        "남자65세부터69세생활인구수",
        "남자70세이상생활인구수",
        "여자0세부터9세생활인구수",
        "여자10세부터14세생활인구수",
        "여자15세부터19세생활인구수",
        "여자20세부터24세생활인구수",
        "여자25세부터29세생활인구수",
        "여자30세부터34세생활인구수",
        "여자35세부터39세생활인구수",
        "여자40세부터44세생활인구수",
        "여자45세부터49세생활인구수",
        "여자50세부터54세생활인구수",
        "여자55세부터59세생활인구수",
        "여자60세부터64세생활인구수",
        "여자65세부터69세생활인구수",
        "여자70세이상생활인구수",
    ]

    df2 = pd.read_csv(
        sample_file,
        encoding="utf-8",
        dtype={"기준일ID": str, "시간대구분": str, "행정동코드": str},
        usecols=expected_cols,
        header=0,
    )
    print(f"   행 수: {len(df2)}")
    print(f"   컬럼 수: {len(df2.columns)}")
    print(f"   첫 5개 컬럼: {df2.columns[:5].tolist()}")
    print(f"   행정동코드 샘플: {df2['행정동코드'].head(3).tolist()}")
    print(f"   총생활인구수 샘플: {df2['총생활인구수'].head(3).tolist()}")

except Exception as e:
    print(f"   에러: {e}")

# 3. 원본 파일 첫 몇 줄 직접 확인
print("\n3. 원본 파일 첫 3줄:")
with open(sample_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 3:
            print(f"   줄 {i+1}: {line.strip()[:100]}...")
        else:
            break

# 2. Trading Area 샘플 확인 (더 자세히)
trading_path = os.path.abspath("../../Data/Trading_Area/Trading_Area_2019.csv")
trading_sample = pd.read_csv(trading_path, encoding="utf-8")
print(f"\nTrading Area 2019년 데이터 ({len(trading_sample)}행):")
print("기준_년분기_코드 고유값:", sorted(trading_sample["기준_년분기_코드"].unique()))

# 각 기준_년분기_코드별 데이터 개수
print("\n각 분기별 데이터 개수:")
print(trading_sample["기준_년분기_코드"].value_counts().sort_index())

# 3. 우리가 만든 LocalPeople 집계 데이터
print("\n우리가 만든 LocalPeople 집계:")
print("20191, 20192, 20193, 20194 형식")
print("→ 이 값들이 Trading Area와 정확히 매치되는지 확인 필요!")
