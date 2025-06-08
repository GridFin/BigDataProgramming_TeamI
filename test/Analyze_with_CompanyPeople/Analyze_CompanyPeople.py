import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
import gc
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")


def load_companypeople_quarterly_fixed(year, quarter, people_dir):
    """수정된 분기별 CompanyPeople 데이터 로드"""

    file_path = os.path.join(people_dir, "CompanyPeople.csv")

    if not os.path.exists(file_path):
        print(f"    ❌ CompanyPeople 파일이 존재하지 않습니다: {file_path}")
        return None

    print(f"    {year}년 {quarter}분기 직장인구 데이터 로드 중...")

    try:
        # CSV 읽기 - euc-kr 인코딩 우선 시도
        try:
            df = pd.read_csv(file_path, encoding="euc-kr")
        except:
            try:
                df = pd.read_csv(file_path, encoding="cp949")
            except:
                df = pd.read_csv(file_path, encoding="utf-8")

        # 해당 년도/분기 데이터 필터링
        quarter_code = int(f"{year}{quarter}")
        df_filtered = df[df["기준_년분기_코드"] == quarter_code].copy()

        if len(df_filtered) == 0:
            print(f"    ❌ {year}년 {quarter}분기 데이터가 없습니다")
            return None

        # 행정동코드 정리
        df_filtered["행정동코드"] = df_filtered["행정동_코드"].astype(str).str.zfill(8)

        # 필요한 컬럼만 선택
        numeric_cols = [
            "총_직장_인구_수",
            "남성_직장_인구_수",
            "여성_직장_인구_수",
            "연령대_10_직장_인구_수",
            "연령대_20_직장_인구_수",
            "연령대_30_직장_인구_수",
            "연령대_40_직장_인구_수",
            "연령대_50_직장_인구_수",
            "연령대_60_이상_직장_인구_수",
            "남성연령대_10_직장_인구_수",
            "남성연령대_20_직장_인구_수",
            "남성연령대_30_직장_인구_수",
            "남성연령대_40_직장_인구_수",
            "남성연령대_50_직장_인구_수",
            "남성연령대_60_이상_직장_인구_수",
            "여성연령대_10_직장_인구_수",
            "여성연령대_20_직장_인구_수",
            "여성연령대_30_직장_인구_수",
            "여성연령대_40_직장_인구_수",
            "여성연령대_50_직장_인구_수",
            "여성연령대_60_이상_직장_인구_수",
        ]

        keep_cols = ["행정동코드"] + [
            col for col in numeric_cols if col in df_filtered.columns
        ]
        result_df = df_filtered[keep_cols].copy()

        print(
            f"    ✅ {year}년 {quarter}분기 직장인구 데이터 로드 완료: {len(result_df)}개 행정동"
        )
        return result_df

    except Exception as e:
        print(f"    ❌ {year}년 {quarter}분기 직장인구 데이터 로드 실패: {e}")
        return None


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
    """수정된 마스터 데이터셋 생성 (CompanyPeople 버전)"""
    print("=== CompanyPeople 마스터 데이터셋 생성 시작 ===")

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

            # CompanyPeople 데이터 로드
            people_quarter = load_companypeople_quarterly_fixed(
                year, quarter, people_dir
            )

            if people_quarter is None:
                print(f"    ❌ {year}년 {quarter}분기 직장인구 데이터 없음")
                continue

            # 데이터 병합
            merged_data = pd.merge(
                biz_quarter, people_quarter, on="행정동코드", how="left"
            )

            # 병합 결과 확인
            people_cols = [col for col in people_quarter.columns if col != "행정동코드"]
            if people_cols:
                people_match_rate = (
                    (~merged_data[people_cols[0]].isna()).sum() / len(merged_data) * 100
                )
            else:
                people_match_rate = 0

            print(
                f"    ✅ 병합 완료: 매출 {len(biz_quarter)} + 직장인구 {len(people_quarter)} → {len(merged_data)}행"
            )
            print(f"    📊 직장인구 데이터 매칭률: {people_match_rate:.1f}%")

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
    """수정된 데이터 전처리 (CompanyPeople 버전)"""
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

    # CompanyPeople 관련 컬럼 확인
    people_cols = [
        col
        for col in df.columns
        if any(
            keyword in col
            for keyword in ["직장_인구_수", "남성_직장", "여성_직장", "연령대_"]
        )
    ]
    print(f"📊 CompanyPeople 관련 컬럼: {len(people_cols)}개")

    if len(people_cols) == 0:
        print("⚠️ 경고: CompanyPeople 데이터가 포함되지 않았습니다!")
    else:
        print("✅ CompanyPeople 데이터 포함 확인됨")
        # 몇 개 컬럼명 출력
        sample_cols = people_cols[:5]
        print(f"   예시: {sample_cols}")

    print(f"전처리 후 데이터 크기: {df.shape}")
    return df


def train_and_evaluate_model_fixed(X_train, y_train, X_test, y_test):
    """수정된 모델 학습 및 평가 (CompanyPeople 버전)"""
    print("\n=== 모델 학습 및 평가 ===")
    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

    # 타겟 변수 로그 변환
    y_train_log = np.log1p(y_train.clip(lower=0))

    # RandomForest 모델
    model = RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, oob_score=True
    )

    print("🚀 모델 학습 시작...")
    model.fit(X_train, y_train_log)
    print("✅ 모델 학습 완료!")

    # 예측
    print("🔮 예측 중...")
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)  # 음수 제거

    # 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n" + "=" * 60)
    print("🎯 CompanyPeople 베이스라인 모델 성능 평가 결과")
    print("=" * 60)
    print(f"MSE:  {mse:,.0f}")
    print(f"RMSE: {rmse:,.0f} 원")
    print(f"MAE:  {mae:,.0f} 원")
    print(f"테스트 데이터 평균 매출: {y_test.mean():,.0f} 원")
    print(f"Out-of-Bag Score: {model.oob_score_:.4f}")

    # 상위 20개 특성 중요도 출력
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\n--- 상위 20개 특성 중요도 ---")
    for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        is_people = any(
            keyword in row["feature"]
            for keyword in ["직장_인구_수", "남성_직장", "여성_직장", "연령대_"]
        )
        marker = "🏢" if is_people else "🏪"
        print(f"{i:2d}. {marker} {row['feature'][:50]}: {row['importance']:.4f}")

    # CompanyPeople 데이터 기여도
    people_features = feature_importance[
        feature_importance["feature"].str.contains(
            "직장_인구_수|남성_직장|여성_직장|연령대_", regex=True
        )
    ]
    people_importance = people_features["importance"].sum()

    print(
        f"\n📊 CompanyPeople 데이터 총 기여도: {people_importance:.4f} ({people_importance*100:.1f}%)"
    )

    if people_importance < 0.05:
        print("⚠️ CompanyPeople 데이터 기여도가 매우 낮습니다.")
    elif people_importance < 0.15:
        print("🔶 CompanyPeople 데이터 기여도가 낮습니다.")
    else:
        print("✅ CompanyPeople 데이터가 유의미하게 기여하고 있습니다.")

    return (
        model,
        {"mse": mse, "rmse": rmse, "mae": mae},
        feature_importance,
        people_importance,
    )


def save_model_and_metrics(
    model, metrics, feature_importance_df, people_importance, config
):
    """모델과 평가지표 저장 (CompanyPeople 버전)"""

    # 결과 저장 디렉토리 생성 (현재 스크립트 디렉토리 기준)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"📁 결과 디렉토리 생성: {results_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 모델 저장
    model_filename = os.path.join(
        results_dir, f"companypeople_model_{timestamp}.joblib"
    )
    joblib.dump(model, model_filename)
    print(f"💾 모델 저장 완료: {model_filename}")

    # 2. 평가지표 CSV 저장
    metrics_data = {
        "실행시간": [timestamp],
        "MSE": [metrics["mse"]],
        "RMSE": [metrics["rmse"]],
        "MAE": [metrics["mae"]],
        "CompanyPeople_기여도": [people_importance],
        "CompanyPeople_기여도_퍼센트": [people_importance * 100],
        "OOB_Score": [model.oob_score_],
        "테스트_년도": [config["test_year"]],
        "훈련_데이터_크기": [config["train_size"]],
        "테스트_데이터_크기": [config["test_size"]],
        "특성_개수": [config["n_features"]],
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_filename = os.path.join(
        results_dir, f"companypeople_metrics_{timestamp}.csv"
    )
    metrics_df.to_csv(metrics_filename, index=False, encoding="utf-8-sig")
    print(f"📊 평가지표 CSV 저장 완료: {metrics_filename}")

    # 3. 특성 중요도 저장
    importance_filename = os.path.join(
        results_dir, f"companypeople_importance_{timestamp}.csv"
    )
    feature_importance_df.to_csv(importance_filename, index=False, encoding="utf-8-sig")
    print(f"🔍 특성 중요도 CSV 저장 완료: {importance_filename}")

    # 4. 실행 정보 요약 저장
    summary_data = {
        "항목": [
            "실행시간",
            "MSE",
            "RMSE (원)",
            "MAE (원)",
            "CompanyPeople 기여도 (%)",
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
        results_dir, f"companypeople_summary_{timestamp}.csv"
    )
    summary_df.to_csv(summary_filename, index=False, encoding="utf-8-sig")
    print(f"📋 모델 요약 CSV 저장 완료: {summary_filename}")

    print(f"\n🎉 모든 결과가 '{results_dir}' 폴더에 저장되었습니다!")
    return {
        "model_file": model_filename,
        "metrics_file": metrics_filename,
        "importance_file": importance_filename,
        "summary_file": summary_filename,
    }


def main():
    """main 함수 (CompanyPeople 버전)"""
    print("🚀 CompanyPeople 상권 매출 예측 베이스라인 모델")
    print("=" * 50)

    # 설정
    YEARS = range(2019, 2025)
    PEOPLE_DIR = "Data/CompanyPeople"
    BIZ_DIR = "Data/Trading_Area"
    TEST_YEAR = 2024

    try:
        # 1. 데이터 로드 및 병합
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

        # 4. 모델 학습 및 평가
        model, metrics, feature_importance, people_importance = (
            train_and_evaluate_model_fixed(
                X_train_filled, y_train, X_test_filled, y_test
            )
        )

        print("\n🎉 CompanyPeople 베이스라인 모델 학습 완료!")
        print("=" * 50)

        # 5. 모델 및 결과 저장
        config = {
            "test_year": TEST_YEAR,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "n_features": len(feature_cols),
        }
        result = save_model_and_metrics(
            model, metrics, feature_importance, people_importance, config
        )

        print("\n🎉 모든 결과가 저장되었습니다!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
