import pandas as pd
import skops.io as skops
from sklearn import impute
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_drop_duplicates(df):
    """Удаляем дубликаты. Если признаковое описание одинаковое,
    а таргеты разные, оставляем первый встретившийся объект"""

    df_no_doubles = df.drop_duplicates(keep="first")
    idx_first_obj = (
        df_no_doubles.drop(columns="selling_price")
        .drop_duplicates(keep="first")
        .index
    )
    no_d_series = pd.Series(
        dict((x, x in idx_first_obj) for x in df_no_doubles.index)
    )
    df = df_no_doubles[no_d_series].copy()
    df = df.reset_index(drop=True)

    return df


def mileage_clean_n_cast(df):
    """Удаляем из признака единицы измерения и приводим типы"""

    df["mileage"] = df["mileage"].map(
        lambda x: float(str(x).rstrip(" kmpl").rstrip(" km/kg"))
    )
    return df


def engine_clean_n_cast(df):
    """Удаляем из признака единицы измерения и приводим типы"""

    df["engine"] = df["engine"].map(lambda x: float(str(x).rstrip(" CC")))
    return df


def max_power_clean_n_cast(df):
    """Удаляем из признака единицы измерения и приводим типы"""

    df["max_power"] = df["max_power"].map(
        lambda x: str(x).rstrip(" bhp").rstrip("bhp")
    )
    # Заменим '' на nan
    df["max_power"] = df["max_power"].replace(
        to_replace="^$", value="nan", regex=True
    )

    df["max_power"] = df["max_power"].astype("float64")
    return df


def filling_na_with_medians(cfg, df, mode):
    """Заполняем пустые значения столбцов в train и test медианами
    этих столбцов из train"""

    mask = df.isna().any()
    X_notna = df[df.columns[~mask].values]
    X_nums_na = df[df.columns[mask].values]

    if mode == "train":
        na_replacer = impute.SimpleImputer(strategy="median")
        na_replacer.fit(X_nums_na)
        skops.dump(na_replacer, cfg["preprocessing"]["imputer_path"])
    elif mode == "infer":
        na_replacer = skops.load(
            cfg["preprocessing"]["imputer_path"], trusted=True
        )
    else:
        raise ValueError('Wrong mode value - requires "train"/"infer"')

    transformed = na_replacer.transform(X_nums_na)
    X_filled = pd.DataFrame(data=transformed, columns=df.columns[mask].values)

    df = pd.concat([X_notna, X_filled], axis=1)
    return df


def normalize_numerical(cfg, X, mode):
    """Стандартизируем вещественные признаки,
    для теста - стандартизируем по трейну"""

    numerical_cols = X.columns[X.dtypes != object]
    X_num = X[numerical_cols].to_numpy()

    if mode == "train":
        normalizer = StandardScaler()
        normalizer.fit(X_num)
        X_num_scaled = normalizer.transform(X_num)
        skops.dump(normalizer, cfg["preprocessing"]["normalizer_path"])
    elif mode == "infer":
        normalizer = skops.load(
            cfg["preprocessing"]["normalizer_path"], trusted=True
        )
        X_num_scaled = normalizer.transform(X_num)
    else:
        raise ValueError('Wrong mode value - requires "train"/"infer"')

    X_num_df = pd.DataFrame(X_num_scaled, columns=numerical_cols)
    return X_num_df


def encoding_categorical(cfg, X, mode):
    """Кодируем категориальные признаки
    с помощью One Hot Encoding"""

    categorial_cols = X.columns[X.dtypes == object]
    X_cat = X[categorial_cols].to_numpy()

    if mode == "train":
        enc = OneHotEncoder()
        enc.fit(X_cat)
        X_cat_coded = enc.transform(X_cat).toarray()
        skops.dump(enc, cfg["preprocessing"]["encoder_path"])

    elif mode == "infer":
        enc = skops.load(
            cfg["preprocessing"]["encoder_path"],
            trusted=True,
        )
        X_cat_coded = enc.transform(X_cat).toarray()

    else:
        raise ValueError('Wrong mode value - requires "train"/"infer"')

    # Уберём лишний столбец после энкодинга
    X_cat_coded_df = pd.DataFrame(
        X_cat_coded, columns=enc.get_feature_names_out()
    )
    X_cat_df = X_cat_coded_df[X_cat_coded_df.columns[:-1]]
    return X_cat_df
