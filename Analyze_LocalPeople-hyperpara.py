import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import warnings
import gc
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")


def load_localpeople_quarterly_fixed(year, quarter, people_dir):
    """수정된 분기별 LocalPeople 데이터 로드"""

    quarter_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}

    months = quarter_months[quarter]
    print(f"    {year}년 {quarter}분기 인구 데이터 로드 중...")

    monthly_agg_list = []

    for month in months:
        month_str = f"{year}{month:02d}"
        file_path = os.path.join(people_dir, f"LOCAL_PEOPLE_DONG_{month_str}.csv")

        if not os.path.exists(file_path):
            continue

        try:
            # chunk 단위로 읽기
            chunk_list = []
            chunksize = 50000

            # 🔥 CSV 읽기 문제 해결: dtype 강제 지정 + BOM 처리
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
                df_iter = pd.read_csv(
                    file_path,
                    encoding="utf-8",
                    chunksize=chunksize,
                    dtype={
                        "기준일ID": str,
                        "시간대구분": str,
                        "행정동코드": str,
                    },  # 핵심: str로 강제
                    usecols=expected_cols,  # 정확한 컬럼만 선택
                    header=0,  # 첫 번째 행이 헤더임을 명시
                )
            except:
                df_iter = pd.read_csv(
                    file_path,
                    encoding="cp949",
                    chunksize=chunksize,
                    dtype={"기준일ID": str, "시간대구분": str, "행정동코드": str},
                    usecols=expected_cols,
                    header=0,
                )

            for chunk in df_iter:
                chunk["행정동코드"] = chunk["행정동코드"].astype(str).str.zfill(8)

                # 필요한 컬럼만 선택
                numeric_cols = ["총생활인구수"] + [
                    col
                    for col in chunk.columns
                    if ("남자" in col or "여자" in col) and "생활인구수" in col
                ]
                keep_cols = ["행정동코드"] + numeric_cols
                chunk = chunk[keep_cols]

                # 행정동별 집계
                chunk_agg = (
                    chunk.groupby("행정동코드").mean(numeric_only=True).reset_index()
                )
                chunk_list.append(chunk_agg)

            # 월별 데이터 합치기
            if chunk_list:
                month_df = pd.concat(chunk_list, ignore_index=True)
                month_agg = (
                    month_df.groupby("행정동코드").mean(numeric_only=True).reset_index()
                )
                monthly_agg_list.append(month_agg)
                print(f"      {month_str} 완료 ({len(month_agg)}개 행정동)")

            del chunk_list
            gc.collect()

        except Exception as e:
            print(f"      {month_str} 로드 실패: {e}")
            continue

    if not monthly_agg_list:
        return None

    # 분기별 평균 계산
    quarter_df = monthly_agg_list[0]
    for month_df in monthly_agg_list[1:]:
        quarter_df = pd.merge(
            quarter_df,
            month_df,
            on="행정동코드",
            how="outer",
            suffixes=("", "_temp"),
        )

        # 평균 계산
        for col in month_df.columns:
            if col != "행정동코드":
                if f"{col}_temp" in quarter_df.columns:
                    quarter_df[col] = quarter_df[[col, f"{col}_temp"]].mean(
                        axis=1, skipna=True
                    )
                    quarter_df = quarter_df.drop(columns=[f"{col}_temp"])

    print(
        f"    {year}년 {quarter}분기 인구 데이터 집계 완료: {len(quarter_df)}개 행정동"
    )
    return quarter_df


def load_trading_area_data(year, biz_dir):
    """Trading Area 데이터 로드"""
    file_path = os.path.join(biz_dir, f"Trading_Area_{year}.csv")

    if not os.path.exists(file_path):
        return None

    try:
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except:
            df = pd.read_csv(file_path, encoding="cp949")

        df = df.rename(columns={"행정동_코드": "행정동코드"})
        df["행정동코드"] = df["행정동코드"].astype(str).str.zfill(8)

        print(f"  Trading Area {year} 로드 완료: {len(df)}행")
        return df

    except Exception as e:
        print(f"  Trading Area {year} 로드 실패: {e}")
        return None


def create_master_dataset_fixed(years, people_dir, biz_dir):
    """수정된 마스터 데이터셋 생성"""
    print("=== 수정된 마스터 데이터셋 생성 시작 ===")

    all_data = []

    for year in years:
        print(f"\n📅 {year}년 데이터 처리 중...")

        # Trading Area 데이터 로드
        biz_df = load_trading_area_data(year, biz_dir)
        if biz_df is None:
            continue

        # 각 분기별로 처리
        for quarter in [1, 2, 3, 4]:
            print(f"  🔄 {year}년 {quarter}분기 처리 중...")

            # 해당 분기 Trading Area 데이터 필터링
            quarter_code = int(f"{year}{quarter}")
            biz_quarter = biz_df[biz_df["기준_년분기_코드"] == quarter_code].copy()

            if len(biz_quarter) == 0:
                print(f"    ❌ {year}년 {quarter}분기 매출 데이터 없음")
                continue

            # LocalPeople 데이터 로드 (수정된 버전)
            people_quarter = load_localpeople_quarterly_fixed(year, quarter, people_dir)

            if people_quarter is None:
                print(f"    ❌ {year}년 {quarter}분기 인구 데이터 없음")
                continue

            # 데이터 병합
            merged_data = pd.merge(
                biz_quarter, people_quarter, on="행정동코드", how="left"
            )

            # 병합 결과 확인 (첫 번째 인구 컬럼으로 매칭률 확인)
            people_cols = [col for col in people_quarter.columns if col != "행정동코드"]
            if people_cols:
                people_match_rate = (
                    (~merged_data[people_cols[0]].isna()).sum() / len(merged_data) * 100
                )
            else:
                people_match_rate = 0

            print(
                f"    ✅ 병합 완료: 매출 {len(biz_quarter)} + 인구 {len(people_quarter)} → {len(merged_data)}행"
            )
            print(f"    📊 인구 데이터 매칭률: {people_match_rate:.1f}%")

            all_data.append(merged_data)

            # 메모리 정리
            del biz_quarter, people_quarter, merged_data
            gc.collect()

    if not all_data:
        raise ValueError("생성된 데이터가 없습니다!")

    # 전체 데이터 결합
    print("\n🔗 전체 데이터 결합 중...")
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ 마스터 데이터셋 생성 완료: {final_df.shape}")

    return final_df


def preprocess_data_fixed(df):
    """수정된 데이터 전처리"""
    print("\n=== 데이터 전처리 시작 ===")
    print(f"원본 데이터 크기: {df.shape}")

    # 타겟 변수 확인
    if "당월_매출_금액" not in df.columns:
        raise ValueError("타겟 변수 '당월_매출_금액'이 없습니다!")

    # 불필요한 컬럼 제거 + 데이터 리키지 방지
    drop_cols = ["행정동_코드_명", "서비스_업종_코드_명"]

    # 🔥 매출 관련 특성 제거 (데이터 리키지 방지)
    sales_cols = [
        col
        for col in df.columns
        if any(
            keyword in col
            for keyword in [
                "매출_금액",
                "매출_건수",
                "월요일_",
                "화요일_",
                "수요일_",
                "목요일_",
                "금요일_",
                "토요일_",
                "일요일_",
                "주중_",
                "주말_",
                "시간대_",
                "남성_매출",
                "여성_매출",
                "연령대_",
            ]
        )
    ]

    # 타겟 변수는 유지
    sales_cols = [col for col in sales_cols if col != "당월_매출_금액"]

    print(f"데이터 리키지 방지를 위해 제거할 매출 관련 컬럼: {len(sales_cols)}개")
    print(f"제거 컬럼 예시: {sales_cols[:5]}")

    drop_cols.extend(sales_cols)
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 결측값이 너무 많은 컬럼 제거 (70% 이상)
    null_ratio = df.isnull().mean()
    high_null_cols = null_ratio[null_ratio > 0.7].index.tolist()
    print(f"결측률 70% 이상 컬럼 제거: {len(high_null_cols)}개")
    if high_null_cols:
        df = df.drop(columns=high_null_cols)

    # 범주형 변수 인코딩
    if "서비스_업종_코드" in df.columns:
        le = LabelEncoder()
        df["서비스_업종_코드_encoded"] = le.fit_transform(
            df["서비스_업종_코드"].astype(str)
        )
        df = df.drop(columns=["서비스_업종_코드"])

    # LocalPeople 관련 컬럼 확인
    people_cols = [
        col
        for col in df.columns
        if any(keyword in col for keyword in ["생활인구수", "남자", "여자"])
    ]
    print(f"📊 LocalPeople 관련 컬럼: {len(people_cols)}개")

    if len(people_cols) == 0:
        print("⚠️ 경고: LocalPeople 데이터가 포함되지 않았습니다!")
    else:
        print("✅ LocalPeople 데이터 포함 확인됨")
        # 몇 개 컬럼명 출력
        sample_cols = people_cols[:5]
        print(f"   예시: {sample_cols}")

    print(f"전처리 후 데이터 크기: {df.shape}")
    return df


def hyperparameter_tuning_and_predict(X_train, y_train, X_test, y_test, X_future=None):
    """하이퍼파라미터 튜닝 및 모델 학습 평가"""
    print("\n=== 하이퍼파라미터 튜닝 시작 ===")
    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

    # 타겟 변수 로그 변환
    y_train_log = np.log1p(y_train.clip(lower=0))
    y_test_log = np.log1p(y_test.clip(lower=0))

    # 하이퍼파라미터 그리드 정의
    param_grid = {
        "n_estimators": [200, 400, 800, 1200],
        "max_depth": [10, 20, 30, None],
        "min_samples_leaf": [1, 3, 10, 30],
        "max_features": ["auto", "sqrt", 0.3, 0.5, 0.8],
        "bootstrap": [True, False],
        "oob_score": [True],  # 고정
        "random_state": [42],  # 고정
        "n_jobs": [-1],  # 고정
    }

    # TimeSeriesSplit 설정 (5-fold)
    tscv = TimeSeriesSplit(n_splits=5)

    # RandomizedSearchCV 설정
    rf_base = RandomForestRegressor()

    print("🔍 RandomizedSearchCV 실행 중...")
    print(f"   - 파라미터 조합 탐색: {50}개")
    print(f"   - CV 폴드: {5}개")
    print(f"   - 평가 지표: neg_mean_absolute_error")

    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_grid,
        n_iter=50,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    # 하이퍼파라미터 튜닝 실행
    random_search.fit(X_train, y_train_log)

    print("✅ 하이퍼파라미터 튜닝 완료!")
    print(f"🏆 베스트 CV 스코어: {-random_search.best_score_:.4f}")
    print("🎯 베스트 파라미터:")
    for param, value in random_search.best_params_.items():
        print(f"   - {param}: {value}")

    # 베스트 모델로 예측
    best_model = random_search.best_estimator_

    print("\n🔮 테스트 데이터 예측 중...")
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)  # 음수 제거

    # 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n" + "=" * 70)
    print("🎯 LocalPeople 하이퍼파라미터 튜닝 모델 성능 평가 결과")
    print("=" * 70)
    print(f"MSE:  {mse:,.0f}")
    print(f"RMSE: {rmse:,.0f} 원")
    print(f"MAE:  {mae:,.0f} 원")
    print(f"테스트 데이터 평균 매출: {y_test.mean():,.0f} 원")
    print(f"베스트 CV MAE: {-random_search.best_score_:,.0f} 원")
    print(f"Out-of-Bag Score: {best_model.oob_score_:.4f}")

    # 특성 중요도 분석
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\n--- 상위 20개 특성 중요도 ---")
    for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        is_people = any(
            keyword in row["feature"] for keyword in ["생활인구수", "남자", "여자"]
        )
        marker = "🏠" if is_people else "🏪"
        print(f"{i:2d}. {marker} {row['feature'][:50]}: {row['importance']:.4f}")

    # LocalPeople 데이터 기여도
    people_features = feature_importance[
        feature_importance["feature"].str.contains("생활인구수|남자|여자", regex=True)
    ]
    people_importance = people_features["importance"].sum()

    print(
        f"\n📊 LocalPeople 데이터 총 기여도: {people_importance:.4f} ({people_importance*100:.1f}%)"
    )

    # 2025년 예측 (만약 데이터가 있다면)
    future_predictions = None
    if X_future is not None and len(X_future) > 0:
        print("\n🔮 2025년 데이터 예측 중...")
        y_future_log = best_model.predict(X_future)
        future_predictions = np.expm1(y_future_log)
        future_predictions = np.clip(future_predictions, 0, None)
        print(f"✅ 2025년 예측 완료: {len(future_predictions)}개 예측값")
        print(f"   평균 예측 매출: {future_predictions.mean():,.0f} 원")

    return (
        best_model,
        {"mse": mse, "rmse": rmse, "mae": mae, "cv_mae": -random_search.best_score_},
        feature_importance,
        people_importance,
        random_search.best_params_,
        future_predictions,
    )


def save_model_and_metrics(
    model,
    metrics,
    feature_importance_df,
    people_importance,
    best_params,
    config,
    future_predictions=None,
    future_data=None,
):
    """모델과 평가지표 저장"""

    # 결과 저장 디렉토리 생성 (현재 스크립트 디렉토리 기준)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"📁 결과 디렉토리 생성: {results_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 모델 저장
    model_filename = os.path.join(
        results_dir, f"localpeople_tuned_model_{timestamp}.joblib"
    )
    joblib.dump(model, model_filename)
    print(f"💾 모델 저장 완료: {model_filename}")

    # 2. 평가지표 CSV 저장
    metrics_data = {
        "실행시간": [timestamp],
        "MSE": [metrics["mse"]],
        "RMSE": [metrics["rmse"]],
        "MAE": [metrics["mae"]],
        "CV_MAE": [metrics["cv_mae"]],
        "LocalPeople_기여도": [people_importance],
        "LocalPeople_기여도_퍼센트": [people_importance * 100],
        "OOB_Score": [model.oob_score_],
        "테스트_년도": [config["test_year"]],
        "훈련_데이터_크기": [config["train_size"]],
        "테스트_데이터_크기": [config["test_size"]],
        "특성_개수": [config["n_features"]],
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_filename = os.path.join(
        results_dir, f"localpeople_tuned_metrics_{timestamp}.csv"
    )
    metrics_df.to_csv(metrics_filename, index=False, encoding="utf-8-sig")
    print(f"📊 평가지표 CSV 저장 완료: {metrics_filename}")

    # 3. 베스트 파라미터 저장
    params_df = pd.DataFrame([best_params])
    params_filename = os.path.join(
        results_dir, f"localpeople_best_params_{timestamp}.csv"
    )
    params_df.to_csv(params_filename, index=False, encoding="utf-8-sig")
    print(f"🎯 베스트 파라미터 CSV 저장 완료: {params_filename}")

    # 4. 특성 중요도 저장
    importance_filename = os.path.join(
        results_dir, f"localpeople_tuned_importance_{timestamp}.csv"
    )
    feature_importance_df.to_csv(importance_filename, index=False, encoding="utf-8-sig")
    print(f"🔍 특성 중요도 CSV 저장 완료: {importance_filename}")

    # 5. 2025년 예측 결과 저장
    if future_predictions is not None and future_data is not None:
        future_df = future_data.copy()
        future_df["예측_매출_금액"] = future_predictions
        future_filename = os.path.join(
            results_dir, f"localpeople_2025_predictions_{timestamp}.csv"
        )
        future_df.to_csv(future_filename, index=False, encoding="utf-8-sig")
        print(f"🔮 2025년 예측 결과 저장 완료: {future_filename}")

    # 6. 실행 정보 요약 저장
    summary_data = {
        "항목": [
            "실행시간",
            "MSE",
            "RMSE (원)",
            "MAE (원)",
            "CV MAE (원)",
            "LocalPeople 기여도 (%)",
            "OOB Score",
            "테스트 년도",
            "훈련 데이터 크기",
            "테스트 데이터 크기",
            "특성 개수",
        ],
        "값": [
            timestamp,
            f"{metrics['mse']:,.0f}",
            f"{metrics['rmse']:,.0f}",
            f"{metrics['mae']:,.0f}",
            f"{metrics['cv_mae']:,.0f}",
            f"{people_importance*100:.1f}%",
            f"{model.oob_score_:.4f}",
            config["test_year"],
            f"{config['train_size']:,}",
            f"{config['test_size']:,}",
            config["n_features"],
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    summary_filename = os.path.join(
        results_dir, f"localpeople_tuned_summary_{timestamp}.csv"
    )
    summary_df.to_csv(summary_filename, index=False, encoding="utf-8-sig")
    print(f"📋 모델 요약 CSV 저장 완료: {summary_filename}")

    print(f"\n🎉 모든 결과가 '{results_dir}' 폴더에 저장되었습니다!")
    return {
        "model_file": model_filename,
        "metrics_file": metrics_filename,
        "params_file": params_filename,
        "importance_file": importance_filename,
        "summary_file": summary_filename,
        "future_file": future_filename if future_predictions is not None else None,
    }


def main():
    """main 함수"""
    print("🚀 LocalPeople 하이퍼파라미터 튜닝 상권 매출 예측 모델")
    print("=" * 60)

    # 설정
    YEARS = range(2019, 2025)
    PEOPLE_DIR = "Data/LocalPeople"
    BIZ_DIR = "Data/Trading_Area"
    TEST_YEAR = 2024
    FUTURE_YEAR = 2025

    try:
        # 1. 2019-2024 데이터 로드 및 병합
        master_df = create_master_dataset_fixed(YEARS, PEOPLE_DIR, BIZ_DIR)

        # 2. 데이터 전처리
        processed_df = preprocess_data_fixed(master_df)

        # 3. 데이터 분할
        print(f"\n=== 데이터 분할 ({TEST_YEAR}년 테스트) ===")
        train_mask = processed_df["기준_년분기_코드"] < (TEST_YEAR * 10 + 1)
        test_mask = processed_df["기준_년분기_코드"] >= (TEST_YEAR * 10 + 1)

        train_df = processed_df[train_mask].copy()
        test_df = processed_df[test_mask].copy()

        print(f"훈련 데이터: {len(train_df)}행")
        print(f"테스트 데이터: {len(test_df)}행")

        # 특성과 타겟 분리
        target_col = "당월_매출_금액"
        feature_cols = [col for col in train_df.columns if col != target_col]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        # 결측값 처리
        print("결측값 처리 중...")
        imputer = SimpleImputer(strategy="mean")
        X_train_filled = pd.DataFrame(
            imputer.fit_transform(X_train), columns=X_train.columns
        )
        X_test_filled = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # 4. 2025년 데이터 준비 (있다면)
        print(f"\n=== {FUTURE_YEAR}년 데이터 확인 ===")
        future_df = None
        X_future_filled = None

        try:
            future_master = create_master_dataset_fixed(
                [FUTURE_YEAR], PEOPLE_DIR, BIZ_DIR
            )
            if future_master is not None and len(future_master) > 0:
                future_processed = preprocess_data_fixed(future_master)

                # 2025년 데이터에서 특성만 추출 (타겟 변수 제외)
                future_features = [
                    col for col in future_processed.columns if col in feature_cols
                ]
                X_future = future_processed[future_features]
                X_future_filled = pd.DataFrame(
                    imputer.transform(X_future), columns=X_future.columns
                )
                future_df = future_processed.copy()

                print(f"✅ {FUTURE_YEAR}년 데이터 로드 완료: {len(X_future_filled)}행")
            else:
                print(f"❌ {FUTURE_YEAR}년 데이터가 없습니다.")
        except Exception as e:
            print(f"❌ {FUTURE_YEAR}년 데이터 로드 실패: {e}")

        # 5. 하이퍼파라미터 튜닝 및 모델 학습
        (
            model,
            metrics,
            feature_importance,
            people_importance,
            best_params,
            future_predictions,
        ) = hyperparameter_tuning_and_predict(
            X_train_filled, y_train, X_test_filled, y_test, X_future_filled
        )

        print("\n🎉 하이퍼파라미터 튜닝 모델 학습 완료!")
        print("=" * 60)

        # 6. 모델 및 결과 저장
        config = {
            "test_year": TEST_YEAR,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "n_features": len(feature_cols),
        }

        result = save_model_and_metrics(
            model,
            metrics,
            feature_importance,
            people_importance,
            best_params,
            config,
            future_predictions,
            future_df,
        )

        print("\n🎉 모든 결과가 저장되었습니다!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
