#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

data_dir = os.path.abspath("../../Data/LocalPeople")
print("데이터 폴더 경로:", data_dir)


def load_localpeople_quarter(year, quarter, data_dir):
    months = {
        1: ["01", "02", "03"],
        2: ["04", "05", "06"],
        3: ["07", "08", "09"],
        4: ["10", "11", "12"],
    }[quarter]

    dfs = []
    for m in months:
        filename = f"LOCAL_PEOPLE_DONG_{year}{m}.csv"
        file = os.path.join(data_dir, filename)

        if not os.path.exists(file):
            print(f"파일 없음: {filename}")
            continue

        try:
            # 🔥 올바른 CSV 읽기 방식 (Analyze_LocalPeople.py 참조)
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

            try:
                df = pd.read_csv(
                    file,
                    encoding="utf-8",
                    dtype={"기준일ID": str, "시간대구분": str, "행정동코드": str},
                    usecols=expected_cols,
                    header=0,
                )
            except:
                df = pd.read_csv(
                    file,
                    encoding="cp949",
                    dtype={"기준일ID": str, "시간대구분": str, "행정동코드": str},
                    usecols=expected_cols,
                    header=0,
                )

            df["행정동코드"] = df["행정동코드"].astype(str).str.zfill(8)
            selected_cols = ["행정동코드", "총생활인구수"] + [
                col
                for col in df.columns
                if "생활인구수" in col and ("남자" in col or "여자" in col)
            ]
            df = df[selected_cols]
            dfs.append(df)
            print(f"{filename} 로드 완료, {len(df)}행")
        except Exception as e:
            print(f"{filename} 읽기 실패: {e}")

    if not dfs:
        print(f"{year}년 {quarter}분기: 유효한 파일 없음 → 건너뜀")
        return None

    merged = pd.concat(dfs)
    result = merged.groupby("행정동코드").sum().reset_index()
    result["연도"] = year
    result["분기"] = quarter
    result["기준_년분기_코드"] = int(f"{year}{quarter}")
    print(f"{year}년 {quarter}분기 집계 완료, {len(result)}개 행정동")
    return result


all_local = []
for year in range(2019, 2025):
    for quarter in [1, 2, 3, 4]:
        print(f"\n{year}년 {quarter}분기 처리 시작")
        res = load_localpeople_quarter(year, quarter, data_dir)
        if res is not None:
            all_local.append(res)

if all_local:
    local_df = pd.concat(all_local, ignore_index=True)
    print(
        f"\n전체 완료: 총 {len(local_df)}행, {local_df['기준_년분기_코드'].nunique()}개 분기"
    )
else:
    print("\n오류: 로드된 데이터가 없습니다. 경로 또는 파일 누락 여부 확인하세요.")


# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

company_path = os.path.abspath("../../Data/CompanyPeople/CompanyPeople.csv")
trading_dir = os.path.abspath("../../Data/Trading_Area")

# results 폴더 생성
os.makedirs("results", exist_ok=True)

# 직장인 데이터 로드 및 전처리
company_df = pd.read_csv(company_path, encoding="euc-kr")
company_df["행정동코드"] = company_df["행정동_코드"].astype(str).str.zfill(8)
company_df["기준_년분기_코드"] = company_df["기준_년분기_코드"].astype(int)


# 병합 함수 정의
def load_trading_area(year, trading_dir):
    file_path = os.path.join(trading_dir, f"Trading_Area_{year}.csv")
    df = pd.read_csv(file_path, encoding="utf-8")
    df["행정동코드"] = df["행정동_코드"].astype(str).str.zfill(8)
    df["기준_년분기_코드"] = df["기준_년분기_코드"].astype(int)
    return df


# 마스터 데이터 생성
all_merged = []

for year in range(2019, 2025):
    trading_df = load_trading_area(year, trading_dir)

    for quarter in [1, 2, 3, 4]:
        quarter_code = int(f"{year}{quarter}")
        print(f"{quarter_code} 병합 중...")

        # 분기별 데이터 필터링
        trade_q = trading_df[trading_df["기준_년분기_코드"] == quarter_code].copy()
        local_q = local_df[local_df["기준_년분기_코드"] == quarter_code].copy()
        comp_q = company_df[company_df["기준_년분기_코드"] == quarter_code].copy()

        if trade_q.empty:
            print(f"매출 데이터 없음: {quarter_code}")
            continue

        # 병합: 매출 + 실거주 (행정동코드 + 분기 기준)
        merged = pd.merge(
            trade_q,
            local_q,
            on=["행정동코드", "기준_년분기_코드"],
            how="left",
        )

        # 병합: + 직장인 (동일하게 행정동코드 + 분기 기준)
        merged = pd.merge(
            merged,
            comp_q,
            on=["행정동코드", "기준_년분기_코드"],
            how="left",
        )

        print(f"→ 병합 완료: {len(merged)}행")
        all_merged.append(merged)

# 마스터 데이터프레임 완성
master_df = pd.concat(all_merged, ignore_index=True)
print(f"\n최종 마스터 데이터셋 크기: {master_df.shape}")


# In[3]:


X_all = master_df.drop(columns=["당월_매출_금액"])
X_all_numeric = X_all.select_dtypes(include=["number"])

print("전체 컬럼 수:", X_all.shape[1])
print("숫자형 컬럼 수:", X_all_numeric.shape[1])
print("숫자형 컬럼 이름:\n", list(X_all_numeric.columns))


# In[4]:


non_numeric_cols = X_all.select_dtypes(exclude=["number"]).columns.tolist()
numeric_cols = X_all.select_dtypes(include=["number"]).columns.tolist()

print("숫자형 컬럼 수:", len(numeric_cols))
print("비숫자형 컬럼 수:", len(non_numeric_cols))
print("비숫자형 컬럼 목록:\n", non_numeric_cols)


# In[5]:


def preprocess_master(df, test_year=2024):
    df = df.copy(deep=False)

    drop_cols = [
        "행정동_코드_명",
        "서비스_업종_코드_명",
        "행정동명",
        "행정동_코드_명_x",
        "행정동_코드_명_y",
    ]
    leakage_cols = [
        col for col in df.columns if "매출" in col and col != "당월_매출_금액"
    ]
    df.drop(
        columns=[col for col in drop_cols + leakage_cols if col in df.columns],
        inplace=True,
    )

    if "서비스_업종_코드" in df.columns:
        le = LabelEncoder()
        df["업종코드_encoded"] = le.fit_transform(df["서비스_업종_코드"].astype(str))
        df.drop(columns=["서비스_업종_코드"], inplace=True)
        joblib.dump(le, "results/label_encoder.joblib")

    y_all = df["당월_매출_금액"].reset_index(drop=True)
    quarter_col = df["기준_년분기_코드"].reset_index(drop=True)

    X_all = df.drop(columns=["당월_매출_금액"])
    X_all_numeric = X_all.select_dtypes(include=["number"]).copy()

    # ──────────────────────── Imputer: fit on train only ────────────────────────
    train_mask = quarter_col < test_year * 10 + 1
    test_mask = ~train_mask

    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_all_numeric[train_mask]),
        columns=X_all_numeric.columns,
        index=X_all_numeric[train_mask].index,
    )
    X_test = pd.DataFrame(
        imputer.transform(X_all_numeric[test_mask]),
        columns=X_all_numeric.columns,
        index=X_all_numeric[test_mask].index,
    )
    joblib.dump(imputer, "results/imputer.joblib")

    y_train = y_all[train_mask]
    y_test = y_all[test_mask]

    print(
        f"학습용: {len(X_train)}행 / 검증용: {len(X_test)}행 / 특성 수: {X_train.shape[1]}"
    )
    return X_train, X_test, y_train, y_test, quarter_col[test_mask]


# In[6]:


X_train, X_test, y_train, y_test, quarter_test = preprocess_master(master_df)


# In[7]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 로그 변환하여 학습
y_train_log = np.log1p(y_train.clip(lower=0))

# 모델 정의
model = RandomForestRegressor(
    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, oob_score=True
)

print("모델 학습 중...")
model.fit(X_train, y_train_log)
print("학습 완료")

# 예측 및 로그 역변환
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log).clip(min=0)

# 평가 지표 계산
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("\n모델 평가 결과")
print(f"MSE:  {mse:,.0f}")
print(f"RMSE: {rmse:,.0f} 원")
print(f"MAE:  {mae:,.0f} 원")
print(f"OOB Score: {model.oob_score_:.4f}")


# In[8]:


# 예측 결과 요약을 위한 메타 정보 추출
meta_cols = ["기준_년분기_코드", "행정동코드", "행정동_코드_명_x", "서비스_업종_코드"]
test_meta = master_df[master_df["기준_년분기_코드"].isin(quarter_test)].copy()
test_meta = test_meta[meta_cols].reset_index(drop=True)

# 예측 결과 결합
test_meta["예측_매출"] = y_pred
test_meta["실제_매출"] = y_test.reset_index(drop=True)
test_meta["연도"] = test_meta["기준_년분기_코드"] // 10

# 연도+동+업종 단위로 집계
summary_2024 = (
    test_meta.groupby(["연도", "행정동코드", "행정동_코드_명_x", "서비스_업종_코드"])
    .agg(총매출=("실제_매출", "sum"), 예측_총매출=("예측_매출", "sum"))
    .reset_index()
)

# 순위: 행정동별 업종 순위
summary_2024["순위"] = summary_2024.groupby(["연도", "행정동코드"])["예측_총매출"].rank(
    ascending=False, method="min"
)

# 결과 정리
df_2024_result = summary_2024.rename(
    columns={"행정동_코드_명_x": "행정동명", "서비스_업종_코드": "업종"}
)[["연도", "행정동코드", "행정동명", "업종", "예측_총매출", "순위"]]


print(df_2024_result.head())


# In[9]:


# master_df에서 업종 코드 ↔ 업종명 맵핑 추출
업종코드_매핑 = master_df[["서비스_업종_코드", "서비스_업종_코드_명"]].drop_duplicates()

# df_2024_result에 업종명 붙이기
df_2024_result = pd.merge(
    df_2024_result,
    업종코드_매핑,
    how="left",
    left_on="업종",
    right_on="서비스_업종_코드",
).drop(columns=["서비스_업종_코드"])

# 컬럼명 정리
df_2024_result = df_2024_result.rename(columns={"서비스_업종_코드_명": "업종명"})


# In[10]:


# 1. 2019~2023 실제 매출 요약
train_meta_cols = [
    "기준_년분기_코드",
    "행정동코드",
    "행정동_코드_명_x",
    "서비스_업종_코드",
    "당월_매출_금액",
]
train_meta = master_df[master_df["기준_년분기_코드"] < 20241].copy()
train_meta = train_meta[train_meta_cols].copy()
train_meta["연도"] = train_meta["기준_년분기_코드"] // 10

summary_2019_2023 = (
    train_meta.groupby(["연도", "행정동코드", "행정동_코드_명_x", "서비스_업종_코드"])
    .agg(총매출=("당월_매출_금액", "sum"))
    .reset_index()
)

summary_2019_2023 = summary_2019_2023.rename(
    columns={"행정동_코드_명_x": "행정동명", "서비스_업종_코드": "업종"}
)

# 2. 업종명 붙이기
summary_2019_2023 = pd.merge(
    summary_2019_2023,
    업종코드_매핑,
    how="left",
    left_on="업종",
    right_on="서비스_업종_코드",
).drop(columns=["서비스_업종_코드"])

summary_2019_2023 = summary_2019_2023.rename(columns={"서비스_업종_코드_명": "업종명"})

# 3. 행정동 내 업종별 순위 계산
summary_2019_2023["순위"] = summary_2019_2023.groupby(["연도", "행정동코드"])[
    "총매출"
].rank(ascending=False, method="min")

# 4. 2024 예측 결과 포맷 맞추기
df_2024_actual = df_2024_result.rename(columns={"예측_총매출": "총매출"})

# 5. 2019~2024 통합
df_2019_2024 = pd.concat([summary_2019_2023, df_2024_actual], ignore_index=True)

# 6. 확인 + 연도 정수 처리
df_2019_2024["연도"] = df_2019_2024["연도"].astype(int)
df_2019_2024.head()


# In[11]:


# 연도 순으로 정렬 후 저장
df_2019_2024_sorted = df_2019_2024.sort_values(by=["연도", "행정동코드", "순위"])
df_2019_2024_sorted.to_csv("analysis_2019_2024.csv", index=False, encoding="utf-8-sig")


# In[12]:


# 1. 2023~2024년 데이터 기반
df_recent = master_df[master_df["기준_년분기_코드"] >= 20231].copy()

# 2. 저장된 LabelEncoder 로드 (리키지 방지)
le = joblib.load("results/label_encoder.joblib")
df_recent["업종코드_encoded"] = le.transform(df_recent["서비스_업종_코드"].astype(str))

# 3. 기본 메타 정보: 행정동 + 업종 조합
meta_cols = ["행정동코드", "행정동_코드_명_x", "서비스_업종_코드"]
df_meta = df_recent[meta_cols].drop_duplicates()

# 4. 입력 특성 평균값 계산
feature_cols = X_train.columns.tolist()
X_recent = df_recent[feature_cols + meta_cols].copy()
X_avg = (
    X_recent.groupby(["행정동코드", "서비스_업종_코드"])[feature_cols]
    .mean()
    .reset_index()
)

# 5. 메타정보와 평균값 병합 → X_2025 구성
df_2025_input = pd.merge(
    df_meta, X_avg, on=["행정동코드", "서비스_업종_코드"], how="inner"
)
df_2025_input["연도"] = 2025

# 6. 예측
X_2025 = df_2025_input[feature_cols]
y_2025_log = model.predict(X_2025)
y_2025 = np.expm1(y_2025_log).clip(min=0)

# 7. 결과 구성
df_2025_result = df_2025_input[
    ["연도", "행정동코드", "행정동_코드_명_x", "서비스_업종_코드"]
].copy()
df_2025_result["예측_총매출"] = y_2025

# 8. 업종명 붙이기 (서비스_업종_코드 유지하면서 merge)
df_2025_result = pd.merge(
    df_2025_result,
    업종코드_매핑,
    how="left",
    left_on="서비스_업종_코드",
    right_on="서비스_업종_코드",
)

# 9. 컬럼 정리 및 순위 계산
df_2025_result = df_2025_result.rename(
    columns={
        "행정동_코드_명_x": "행정동명",
        "서비스_업종_코드": "업종",
        "서비스_업종_코드_명": "업종명",
    }
)

df_2025_result["순위"] = df_2025_result.groupby("행정동코드")["예측_총매출"].rank(
    ascending=False, method="min"
)

df_2025_result = df_2025_result[
    ["연도", "행정동코드", "행정동명", "업종", "업종명", "예측_총매출", "순위"]
]

# 10. 저장
df_2025_sorted = df_2025_result.sort_values(by=["행정동코드", "순위"])
df_2025_sorted.to_csv("prediction_2025.csv", index=False, encoding="utf-8-sig")


# 확인
df_2025_result.head()
