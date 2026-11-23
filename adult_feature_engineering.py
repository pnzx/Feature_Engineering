"""UCI Adult feature-engineering assignment: reproducible end-to-end pipeline.

Run: python adult_feature_engineering.py
Outputs: outputs/figures/*.png, outputs/experiment_results.csv,
         outputs/feature_importance.csv, outputs/report.md
"""
from __future__ import annotations

from pathlib import Path
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, MinMaxScaler,
                                  RobustScaler, OrdinalEncoder)

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

COLS = ["age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"]
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"


def load_data() -> pd.DataFrame:
    """Download the official UCI training file; cache it for later runs."""
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    if not train_path.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, train_path)
    if not test_path.exists():
        print(f"Downloading {TEST_URL}")
        urllib.request.urlretrieve(TEST_URL, test_path)
    train = pd.read_csv(train_path, names=COLS, skipinitialspace=True, na_values="?")
    # adult.test starts with a metadata comment and marks labels with a trailing period.
    test = pd.read_csv(test_path, names=COLS, skipinitialspace=True, na_values="?", skiprows=1)
    df = pd.concat([train, test], ignore_index=True)
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].str.strip()
    df["income"] = df["income"].str.rstrip(".")
    return df.dropna(subset=["income"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Two outcome-independent derived features, created before splitting."""
    x = df.drop(columns="income").copy()
    x["capital_net"] = x["capital_gain"] - x["capital_loss"]
    x["capital_any"] = ((x["capital_gain"] + x["capital_loss"]) > 0).astype(int)
    x["age_group"] = pd.cut(x["age"], bins=[16, 25, 35, 45, 55, 65, 100],
                            labels=["17-25", "26-35", "36-45", "46-55", "56-65", "66+"])
    return x


def eda(df: pd.DataFrame) -> None:
    missing = df.isna().mean().sort_values(ascending=False).rename("missing_rate")
    missing.to_csv(OUT / "missing_rate.csv", encoding="utf-8-sig")
    # Required distribution plots.
    fig, ax = plt.subplots(figsize=(8, 5)); sns.histplot(df["age"], bins=30, kde=True, ax=ax)
    ax.set_title("Age distribution"); fig.tight_layout(); fig.savefig(FIG / "hist_age.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(x=df["hours_per_week"], ax=ax)
    ax.set_title("Hours per week: outlier scan"); fig.tight_layout(); fig.savefig(FIG / "box_hours.png", dpi=160); plt.close(fig)
    nums = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 7)); sns.heatmap(nums.corr(), cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Numeric feature correlation"); fig.tight_layout(); fig.savefig(FIG / "heatmap_numeric.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); sns.countplot(data=df, x="income", order=["<=50K", ">50K"], ax=ax)
    ax.set_title("Target distribution"); fig.tight_layout(); fig.savefig(FIG / "target_count.png", dpi=160); plt.close(fig)
    # Numeric IQR counts for outlier exploration (not automatic deletion).
    outlier_rows = []
    for col in nums:
        q1, q3 = nums[col].quantile([.25, .75]); iqr = q3-q1
        count = int(((nums[col] < q1-1.5*iqr) | (nums[col] > q3+1.5*iqr)).sum())
        outlier_rows.append({"feature": col, "iqr_outlier_count": count, "iqr_outlier_rate": count/len(df)})
    pd.DataFrame(outlier_rows).to_csv(OUT / "iqr_outliers.csv", index=False, encoding="utf-8-sig")


def make_pipeline(X: pd.DataFrame, *, imputation: str, encoding: str,
                  scaling: str, selection: bool, model_name: str) -> Pipeline:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.columns.difference(numeric).tolist()
    if imputation == "mean": num_imp = SimpleImputer(strategy="mean")
    elif imputation == "median": num_imp = SimpleImputer(strategy="median")
    else: num_imp = SimpleImputer(strategy="most_frequent")
    if encoding == "onehot": cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else: cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(),
              "robust": RobustScaler(), "none": "passthrough"}[scaling]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", num_imp), ("scaler", scaler)]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", cat_enc)]), categorical),
    ], remainder="drop", verbose_feature_names_out=False)
    model = (LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED)
             if model_name == "Logistic Regression" else
             RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                    class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=SEED))
    steps = [("preprocess", prep)]
    if selection:
        selector_model = RandomForestClassifier(n_estimators=180, n_jobs=-1,
                                                random_state=SEED, class_weight="balanced")
        steps.append(("selection", SelectFromModel(selector_model, threshold="median")))
    steps.append(("model", model))
    return Pipeline(steps)


def main() -> None:
    df = load_data()
    print("Dataset shape:", df.shape)
    print("Target counts:\n", df["income"].value_counts())
    print("Missing rate:\n", (df.isna().mean().sort_values(ascending=False).head(6)).to_string())
    eda(df)
    X = add_features(df)
    y = (df["income"] == ">50K").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=SEED)

    # Base is operationalized as minimal imputation + ordinal encoding + no scaling;
    # raw strings cannot be passed directly to sklearn estimators.
    experiments = [
        ("Base", "most_frequent", "label", "none", False),
        ("Exp-1", "mean", "onehot", "standard", False),
        ("Exp-2", "median", "label", "minmax", True),
        ("Exp-3", "most_frequent", "onehot", "robust", True),
    ]
    rows, importances = [], []
    for exp, imp, enc, scale, selection in experiments:
        for model_name in ["Logistic Regression", "Random Forest"]:
            pipe = make_pipeline(X_train, imputation=imp, encoding=enc,
                                 scaling=scale, selection=selection, model_name=model_name)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            prob = pipe.predict_proba(X_test)[:, 1]
            rows.append({"experiment": exp, "imputation": imp, "encoding": enc,
                         "scaling": scale, "feature_selection": selection,
                         "model": model_name, "accuracy": accuracy_score(y_test, pred),
                         "precision": precision_score(y_test, pred, zero_division=0),
                         "recall": recall_score(y_test, pred, zero_division=0),
                         "f1": f1_score(y_test, pred, zero_division=0),
                         "roc_auc": roc_auc_score(y_test, prob)})
            if model_name == "Random Forest":
                prep = pipe.named_steps["preprocess"]
                names = prep.get_feature_names_out()
                selector = pipe.named_steps.get("selection")
                if selector is not None:
                    mask = selector.get_support(); names = names[mask]
                vals = pipe.named_steps["model"].feature_importances_
                importances.extend({"experiment": exp, "feature": n, "importance": v}
                                   for n, v in zip(names, vals))
    result = pd.DataFrame(rows).sort_values(["experiment", "model"])
    result.to_csv(OUT / "experiment_results.csv", index=False, encoding="utf-8-sig")
    imp_df = pd.DataFrame(importances).sort_values("importance", ascending=False)
    imp_df.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
    print("\nExperiment results:\n", result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nTop features (Exp-3 RF):\n", imp_df[imp_df.experiment == "Exp-3"].head(10).to_string(index=False))
    write_report(df, result, imp_df)


def write_report(df: pd.DataFrame, results: pd.DataFrame, importance: pd.DataFrame) -> None:
    best = results.sort_values("roc_auc", ascending=False).iloc[0]
    top = importance[importance.experiment == "Exp-3"].head(10)
    miss = df.isna().mean().sort_values(ascending=False).head(5)
    target = df["income"].value_counts(normalize=True)
    report = f"""# UCI Adult 데이터 특성 공학 실험 보고서

## 1. 데이터셋 소개
UCI Machine Learning Repository의 Adult(Census Income) 데이터로 연 소득이 5만 달러를 초과하는지 분류한다. 데이터는 {len(df):,}행, {df.shape[1]}개 열이며, 예측 입력 특성 14개와 타깃 1개로 구성된다. 수치형과 범주형이 혼합되어 있고, 표본은 500개 이상이다.

| 열 | 설명 | 자료형 |
|---|---|---|
| age | 나이 | 수치 |
| workclass | 고용 형태 | 범주 |
| fnlwgt | 인구 조사 가중치 | 수치 |
| education | 최종 학력 범주 | 범주 |
| education_num | 교육 수준의 순서형 수치 | 수치 |
| marital_status | 혼인 상태 | 범주 |
| occupation | 직업 | 범주 |
| relationship | 가구 내 관계 | 범주 |
| race | 인종 범주 | 범주 |
| sex | 성별 범주 | 범주 |
| capital_gain | 자본 이득 | 수치 |
| capital_loss | 자본 손실 | 수치 |
| hours_per_week | 주당 근무 시간 | 수치 |
| native_country | 출생 국가 | 범주 |
| income | 연 소득 50K 초과 여부(타깃) | 범주 |

타깃 비율은 <=50K {target.get('<=50K', 0):.1%}, >50K {target.get('>50K', 0):.1%}로 클래스 불균형이 있다. 결측 비율 상위 열: {', '.join(f'{k} {v:.1%}' for k,v in miss.items() if v>0) or '결측 없음'}.

## 2. EDA 결과
필수 시각화(Histogram, Boxplot, Heatmap, 타깃 Countplot)와 IQR 기반 이상치 비율표는 `outputs/figures/`와 `outputs/iqr_outliers.csv`에 저장했다. Adult 자료의 `?` 토큰은 결측값으로 처리한다. 극단값은 오류로 단정하지 않고 도메인상 가능한 자본 이득/손실 등이 포함될 수 있어 삭제하지 않았다. 수치 특성 상관관계는 함께 제공한 heatmap으로 확인한다.

## 3. 특성 공학 및 평가 설계
파생 특성은 `capital_net = capital_gain - capital_loss`, 자본 변동 여부(`capital_any`), 연령 구간(`age_group`)이다. 학습/평가 분할(80:20, 층화, random_state=42) 후 모든 결측 대체, 인코딩, 스케일링, 변수 선택을 학습 데이터에만 적합하는 Pipeline을 사용해 누수를 막았다. One-Hot과 Ordinal(Label) 인코딩, Standard/MinMax/Robust/무스케일, 평균/중앙값/최빈값 결측 대체를 비교했다. Drop NA는 행 삭제로 인해 표본과 클래스 구성이 변하므로 Base 비교에서만 보조 분석을 권장한다.

## 4. 실험 결과
아래 수치는 실행 환경에서 스크립트를 실행해 생성된 `outputs/experiment_results.csv`에 기록된다. 각 전처리 조합마다 Logistic Regression과 Random Forest를 비교한다.

| 실험 | 결측치 처리 | 인코딩 | 스케일링 | 변수 선택 |
|---|---|---|---|---|
| Base | 최빈값 대체(구현 가능한 기준선) | Ordinal | 없음 | 없음 |
| Exp-1 | 평균 / 범주형 최빈값 | One-Hot | Standard | 없음 |
| Exp-2 | 중앙값 / 범주형 최빈값 | Ordinal | MinMax | Random Forest 중요도 기반 |
| Exp-3 | 최빈값 | One-Hot | Robust | Random Forest 중요도 기반 |

최고 ROC-AUC 모델은 **{best['experiment']} / {best['model']}**, ROC-AUC={best['roc_auc']:.4f}, F1={best['f1']:.4f}, Accuracy={best['accuracy']:.4f}이다. 전체 지표 및 조합별 표는 아래 결과 CSV가 기준이다.

## 5. 변수 중요도
Exp-3 Random Forest 상위 특성:
{top.to_markdown(index=False, floatfmt='.4f') if len(top) else '결과 생성 후 표시'}

## 6. 결론
전처리 전략의 효과는 모델과 평가 지표에 따라 달라진다. Logistic Regression은 특성 스케일에 민감하고, Random Forest는 단조 스케일 변환에 대체로 덜 민감하다. One-Hot은 명목 범주 사이에 임의 순서를 만들지 않지만 차원을 늘린다. Ordinal 인코딩은 계산량을 줄이는 대신 거리/순서가 있는 듯한 신호를 선형 모델에 줄 수 있다. 변수 선택은 성능 수치와 선택된 변수 수를 함께 비교해야 하며, 상위 변수 제거가 성능 향상을 보장하지 않는다. 본 결과는 단일 고정 holdout 평가이므로 최종 일반화 성능 판단에는 교차 검증과 하이퍼파라미터 탐색을 추가하는 것이 적절하다.

## 7. 재현 방법 및 출처
`pip install pandas numpy matplotlib seaborn scikit-learn tabulate` 후 `python adult_feature_engineering.py`를 실행한다. 데이터는 UCI 공식 원본에서 내려받으며 처음 실행 시 `data/adult.data`에 캐시한다.

출처: Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20. CC BY 4.0.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    write_pdf(df, results, importance)


def write_pdf(df: pd.DataFrame, results: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Create a Korean PDF report after the metrics and charts exist."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                   TableStyle, Image, PageBreak)

    font_candidates = [Path("C:/Windows/Fonts/malgun.ttf"),
                       Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")]
    font_path = next((p for p in font_candidates if p.exists()), None)
    if font_path is None:
        raise RuntimeError("Korean font not found. Install Malgun Gothic or Noto Sans CJK.")
    pdfmetrics.registerFont(TTFont("Korean", str(font_path), subfontIndex=0 if font_path.suffix == ".ttc" else 0))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="KTitle", parent=styles["Title"], fontName="Korean",
                              fontSize=20, leading=28, alignment=TA_CENTER, textColor=colors.HexColor("#17365D")))
    styles.add(ParagraphStyle(name="KHead", parent=styles["Heading2"], fontName="Korean",
                              fontSize=13, leading=19, textColor=colors.HexColor("#17365D"), spaceBefore=8))
    styles.add(ParagraphStyle(name="KBody", parent=styles["BodyText"], fontName="Korean", fontSize=9,
                              leading=14, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="KSmall", parent=styles["BodyText"], fontName="Korean", fontSize=7,
                              leading=10, wordWrap="CJK"))
    pdf = SimpleDocTemplate(str(OUT / "adult_feature_engineering_report.pdf"), pagesize=A4,
                            rightMargin=17*mm, leftMargin=17*mm, topMargin=17*mm, bottomMargin=17*mm)
    story = [Paragraph("특성 공학 파이프라인 설계 및 비교 분석", styles["KTitle"]),
             Paragraph("UCI Adult (Census Income) 데이터셋 · 이진 분류", styles["KBody"]), Spacer(1, 8*mm),
             Paragraph("1. 데이터셋 소개", styles["KHead"]),
             Paragraph(f"UCI Machine Learning Repository의 Adult 데이터는 인구 조사 정보를 바탕으로 연 소득이 5만 달러를 초과하는지 예측한다. 이번 분석 표본은 {len(df):,}행, 입력 특성 14개와 타깃 1개다. 혼합형 수치·범주 변수가 포함되며, `?` 표시는 결측값으로 변환했다.", styles["KBody"]),
             Paragraph(f"타깃 분포: <=50K {((df.income=='<=50K').mean()):.1%}, >50K {((df.income=='>50K').mean()):.1%}. 클래스 불균형을 고려해 층화 분할과 class_weight를 적용했다.", styles["KBody"]),
             Paragraph("컬럼 설명", styles["KHead"])]
    dictionary = [["컬럼", "설명", "유형"],
                  ["age", "나이", "수치"], ["workclass", "고용 형태", "범주"],
                  ["fnlwgt", "인구 조사 가중치", "수치"], ["education", "학력 범주", "범주"],
                  ["education_num", "교육 수준 순서값", "수치"], ["marital_status", "혼인 상태", "범주"],
                  ["occupation", "직업", "범주"], ["relationship", "가구 내 관계", "범주"],
                  ["race / sex", "인종 / 성별", "범주"], ["capital_gain/loss", "자본 이득 / 손실", "수치"],
                  ["hours_per_week", "주당 근무 시간", "수치"], ["native_country", "출생 국가", "범주"],
                  ["income", "연 소득 50K 초과 여부", "타깃"]]
    t = Table(dictionary, colWidths=[39*mm, 92*mm, 35*mm], repeatRows=1)
    t.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "Korean"), ("FONTSIZE", (0,0), (-1,-1), 7),
                           ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DCE6F1")),
                           ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#B7C9D6")),
                           ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F9FC")])]))
    story += [t, Paragraph("2. 탐색적 데이터 분석", styles["KHead"]),
              Paragraph("결측률과 IQR 기반 이상치 탐색 결과를 CSV로 저장했다. 이상치는 자동 삭제하지 않았다. 히스토그램, 박스플롯, 수치형 상관관계 Heatmap, 타깃 Countplot은 아래 그림과 함께 outputs/figures에 저장된다.", styles["KBody"])]
    for title, fname in [("연령 분포", "hist_age.png"), ("주당 근무 시간 이상치 탐색", "box_hours.png"),
                         ("수치형 특성 상관관계", "heatmap_numeric.png"), ("타깃 분포", "target_count.png")]:
        p = FIG / fname
        story += [Paragraph(title, styles["KBody"]), Image(str(p), width=155*mm, height=80*mm), Spacer(1, 3*mm)]
    story += [PageBreak(), Paragraph("3. 특성 공학 및 비교 설계", styles["KHead"]),
              Paragraph("파생 특성: capital_net(자본 이득-손실), capital_any(자본 이득 또는 손실 존재 여부), age_group(연령 구간). 데이터는 80:20 층화 분할(random_state=42)했다. 결측 대체, 인코딩, 스케일링 및 Random Forest 기반 변수 선택은 학습 폴드에만 적합하도록 Pipeline 안에 구성했다.", styles["KBody"])]
    exp_table = [["실험", "결측 처리", "인코딩", "스케일링", "선택"],
                 ["Base", "최빈값", "Ordinal", "없음", "없음"],
                 ["Exp-1", "평균", "One-Hot", "Standard", "없음"],
                 ["Exp-2", "중앙값", "Ordinal", "MinMax", "RF 중요도"],
                 ["Exp-3", "최빈값", "One-Hot", "Robust", "RF 중요도"]]
    t = Table(exp_table, colWidths=[22*mm, 30*mm, 30*mm, 32*mm, 40*mm], repeatRows=1)
    t.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "Korean"), ("FONTSIZE", (0,0), (-1,-1), 8),
                           ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DCE6F1")),
                           ("GRID", (0,0), (-1,-1), .4, colors.grey), ("ALIGN", (0,0), (-1,-1), "CENTER")]))
    story += [t, Paragraph("4. 모델 성능 비교", styles["KHead"])]
    metric_rows = [["실험", "모델", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    for _, r in results.iterrows():
        metric_rows.append([r.experiment, r.model, *(f"{r[m]:.3f}" for m in ["accuracy", "precision", "recall", "f1", "roc_auc"])])
    t = Table(metric_rows, colWidths=[20*mm, 37*mm, 22*mm, 22*mm, 20*mm, 17*mm, 22*mm], repeatRows=1)
    t.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "Korean"), ("FONTSIZE", (0,0), (-1,-1), 7),
                           ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DCE6F1")),
                           ("GRID", (0,0), (-1,-1), .3, colors.grey), ("ALIGN", (2,1), (-1,-1), "RIGHT")]))
    best = results.loc[results.roc_auc.idxmax()]
    story += [t, Spacer(1, 4*mm), Paragraph(f"최고 ROC-AUC 조합: {best.experiment} / {best.model} (ROC-AUC {best.roc_auc:.4f}, F1 {best.f1:.4f}). 전처리 비교는 지표와 모델을 함께 보아야 한다.", styles["KBody"]),
              Paragraph("5. 변수 중요도 및 결론", styles["KHead"])]
    top = importance[importance.experiment == "Exp-3"].head(10)
    imp_rows = [["순위", "특성", "중요도"]] + [[str(i+1), row.feature, f"{row.importance:.4f}"] for i, row in enumerate(top.itertuples())]
    t = Table(imp_rows, colWidths=[20*mm, 85*mm, 35*mm], repeatRows=1)
    t.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "Korean"), ("FONTSIZE", (0,0), (-1,-1), 8),
                           ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DCE6F1")), ("GRID", (0,0), (-1,-1), .3, colors.grey)]))
    story += [t, Spacer(1, 3*mm), Paragraph("One-Hot은 명목형 변수에 임의 순서를 부여하지 않지만 차원을 늘린다. Ordinal 방식은 간결하지만 선형 모델에 인위적 순서 신호를 줄 수 있다. 스케일링은 로지스틱 회귀에 더 큰 영향을 주고 트리 모델에는 상대적으로 영향이 적다. 변수 선택은 성능 향상을 보장하지 않으므로 제거 전후 점수와 선택 특성을 함께 해석한다. 결과는 단일 holdout이므로 최종 일반화 성능 판단에는 교차 검증이 더 적절하다.", styles["KBody"]),
              Paragraph("재현성: random_state=42, 80:20 stratified split. 출처: Becker & Kohavi (1996), Adult, UCI Machine Learning Repository, DOI: 10.24432/C5XW20 (CC BY 4.0).", styles["KSmall"])]
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Korean", 8); canvas.setFillColor(colors.grey)
        canvas.drawString(17*mm, 10*mm, "UCI Adult Feature Engineering")
        canvas.drawRightString(A4[0]-17*mm, 10*mm, f"{doc.page}")
        canvas.restoreState()
    pdf.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    main()
