"""Reproduce the Adult income feature engineering tables with Python 3.12, NumPy and pandas."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import urllib.request
import numpy as np
import pandas as pd
from models import LogisticRegression, RandomForest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
COLS = "age workclass fnlwgt education education_num marital_status occupation relationship race sex capital_gain capital_loss hours_per_week native_country income".split()
NUM = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT = [c for c in COLS[:-1] if c not in NUM]
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
DIGEST = "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2"


def load_data():
    path = ROOT / "adult-all.csv"
    raw = path.read_bytes() if path.exists() else urllib.request.urlopen(DATA_URL, timeout=90).read()
    raw = raw.replace(b"\r\n", b"\n").rstrip(b"\n")
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("Adult mirror did not match the recorded SHA-256 checksum")
    if not path.exists():
        path.write_bytes(raw)
    df = pd.read_csv(path, names=COLS, na_values="?", skipinitialspace=True)
    assert df.shape == (48842, 15) and set(df.income) == {"<=50K", ">50K"}
    return df


@dataclass(frozen=True)
class Spec:
    number: str = "mean"
    category: str = "mode"
    encoding: str = "onehot"
    scale: str = "standard"
    select: bool = False
    derive: bool = True
    base: bool = False


def features(df, spec):
    if spec.base:
        return df[NUM]
    x = df[NUM+CAT].copy()
    if spec.derive:
        x["capital_net"] = x.capital_gain - x.capital_loss
        x["capital_any"] = ((x.capital_gain+x.capital_loss) > 0).astype(int)
        x["age_group"] = pd.cut(x.age, [16,25,35,45,55,65,100],
                                labels=["17-25","26-35","36-45","46-55","56-65","66+"]).astype("str")
    return x


class Preprocessor:
    def __init__(self, spec):
        self.spec = spec

    def fit(self, df, y):
        x = features(df, self.spec)
        self.num = list(x.select_dtypes(include="number"))
        self.cat = [c for c in x if c not in self.num]
        z = x[self.num]
        self.fill = {"mean": z.mean, "median": z.median, "mode": lambda: z.mode().iloc[0]}[self.spec.number]()
        z = z.fillna(self.fill)
        if self.spec.scale == "standard":
            center, scale = z.mean(), z.std(ddof=0)
        elif self.spec.scale == "minmax":
            center, scale = z.min(), z.max()-z.min()
        elif self.spec.scale == "robust":
            center, scale = z.median(), z.quantile(.75)-z.quantile(.25)
        else:
            center, scale = z.iloc[0]*0, z.iloc[0]*0+1
        self.center, self.scale = center, scale.mask(scale == 0, 1)
        self.fills, self.levels = {}, {}
        self.names = self.num.copy()
        for c in self.cat:
            self.fills[c] = x[c].mode().iloc[0] if self.spec.category == "mode" else "__MISSING__"
            self.levels[c] = sorted(x[c].fillna(self.fills[c]).unique())
            self.names.extend([c+"="+v for v in self.levels[c]] if self.spec.encoding == "onehot" else [c])
        a = self.encode(x)
        centered = a - a.mean(axis=0)
        target = y-y.mean()
        denom = np.sqrt((centered**2).sum(axis=0)*np.dot(target, target))
        self.scores = np.divide(np.abs(centered.T@target), denom, out=np.zeros(len(denom)), where=denom>0)
        self.rank = np.argsort(-self.scores, kind="stable")
        self.keep = self.rank[:10] if self.spec.select else np.arange(len(self.names))
        return self

    def encode(self, x):
        z = x[self.num] if self.spec.base else (x[self.num].fillna(self.fill)-self.center)/self.scale
        blocks = [z.to_numpy(dtype=float)]
        for c in self.cat:
            col = x[c].fillna(self.fills[c])
            levels = self.levels[c]
            if self.spec.encoding == "onehot":
                blocks.append(np.column_stack([(col == level).to_numpy(dtype=float) for level in levels]))
            else:
                blocks.append(col.map(dict(zip(levels, range(len(levels))))).fillna(-1).to_numpy(dtype=float)[:, None])
        return np.column_stack(blocks)

    def transform(self, df):
        return self.encode(features(df, self.spec))[:, self.keep]


class Pipeline:
    def __init__(self, spec, estimator):
        self.preprocess, self.estimator = Preprocessor(spec), estimator

    def fit(self, x, y):
        self.preprocess.fit(x, y)
        self.estimator.fit(self.preprocess.transform(x), y)
        return self

    def predict_proba(self, x):
        return self.estimator.predict_proba(self.preprocess.transform(x))


def metrics(y, p):
    pred = p >= .5
    tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), ((~pred) & (y == 1)).sum()
    positives = int(y.sum())
    ranks = pd.Series(p).rank(method="average").to_numpy()
    auc = (ranks[y == 1].sum()-positives*(positives+1)/2)/(positives*(len(y)-positives))
    return dict(accuracy=np.mean(pred == y), precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1),
                f1=2*tp/max(2*tp+fp+fn,1), auc=auc)


def split(df):
    y = (df.income == ">50K").astype(int).to_numpy()
    ids = np.arange(32561)
    rng = np.random.default_rng(42)
    valid = np.sort(np.concatenate([rng.choice(ids[y[ids] == c], round(.2*(y[ids] == c).sum()), replace=False)
                                    for c in (0, 1)]))
    return y, {"train": np.setdiff1d(ids,valid), "validation": valid, "test": np.arange(32561,len(y))}


def main():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    y, subsets = split(df)
    print("Raw shape:", df.shape, "Input shape:", df[COLS[:-1]].shape)
    anchor = Spec()
    exp3 = replace(anchor, number="mode", scale="robust", select=True)
    experiments = {
        "Base": Spec(base=True, derive=False, scale="none"),
        "Exp-1": anchor,
        "Exp-2": replace(anchor, number="median", encoding="ordinal", scale="minmax", select=True),
        "Exp-3": exp3,
        "Ordinal": replace(anchor, encoding="ordinal"),
        "MissingLevel": replace(anchor, category="missing"),
        "MinMax": replace(anchor, scale="minmax"),
        "Robust": replace(anchor, scale="robust"),
        "NoDerived": replace(anchor, derive=False),
        "Exp3-All": replace(exp3, select=False),
    }
    rows, ranks, importances = [], [], []
    for exp, spec in experiments.items():
        for name, model in [("LR", LogisticRegression()), ("RF", RandomForest())]:
            tr = subsets["train"]
            pipe = Pipeline(spec, model).fit(df.iloc[tr], y[tr])
            prep = pipe.preprocess
            for subset, ids in subsets.items():
                scores = metrics(y[ids], pipe.predict_proba(df.iloc[ids]))
                rows.append(dict(experiment=exp, model=name, subset=subset, width=len(prep.keep),
                                 before=len(prep.names), **scores))
            if spec.select and name == "LR":
                for i in prep.keep:
                    ranks.append(dict(experiment=exp, feature=prep.names[i], abs_r=prep.scores[i]))
            if name == "RF":
                for i, score in zip(prep.keep, model.importance):
                    importances.append(dict(experiment=exp, feature=prep.names[i], importance=score))
            print(exp, name, rows[-1]["f1"], flush=True)
    pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(ranks).to_csv(OUT/"selected.csv",index=False)
    pd.DataFrame(importances).to_csv(OUT/"importance.csv",index=False)


if __name__ == "__main__":
    main()
