import random

import hydra
import numpy as np
import pandas as pd
import skops.io as skops
from dvc.api import DVCFileSystem
from hydra.utils import instantiate
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score

import preprocessing as pp


random.seed(42)
np.random.seed(42)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    # Загружаем данные из GDrive
    fs = DVCFileSystem(cfg["source_repo"], cfg["rev"])
    fs.get("data", cfg["data_path"], recursive=True)
    df_train = pd.read_csv(cfg["train"]["df_path"])

    print("Train data shape:", df_train.shape)

    # Оставляем уникальные объекты
    df_train = pp.train_drop_duplicates(df_train)

    # Очищаем столбцы от единиц измерения и приводим к float
    df_train = pp.mileage_clean_n_cast(df_train)
    df_train = pp.engine_clean_n_cast(df_train)
    df_train = pp.max_power_clean_n_cast(df_train)

    # Столбец torque просто дропнем
    df_train = df_train.drop(columns="torque")
    # Заполняем пустые значения медианами
    df_train = pp.filling_na_with_medians(cfg, df_train, cfg["train"]["mode"])

    # Кастуем часть признаков
    df_train["engine"] = df_train["engine"].astype("int64")
    df_train["seats"] = df_train["seats"].astype("int64")

    # вынесем в отдельные переменные таргет
    y_train = df_train["selling_price"].to_numpy()
    X_train = df_train.drop(columns=["selling_price", "name"])

    # стандартизируем вещественные признаки
    X_train_num_scaled = pp.normalize_numerical(
        cfg, X_train, cfg["train"]["mode"]
    )

    # кодируем категориальные признаки
    X_train_cat_coded = pp.encoding_categorical(
        cfg, X_train, cfg["train"]["mode"]
    )

    # объединяем стандартизованный и закодированный элементы
    X_train_preprocessed = X_train_num_scaled.merge(
        X_train_cat_coded, how="inner", left_index=True, right_index=True
    )
    X_train_preprocessed = X_train_preprocessed.to_numpy()

    # Обучение
    model = instantiate(cfg["train"]["model"])
    model.fit(X_train_preprocessed, y_train)
    skops.dump(model, cfg["model_path"])

    # Оценка модели
    y_pred = model.predict(X_train_preprocessed)
    qual_metrics = {
        "r2_train": r2_score(y_train, y_pred),
        "mse_train": MSE(y_train, y_pred),
    }
    [print(f"{k}: {qual_metrics[k]}") for k in qual_metrics]


if __name__ == "__main__":
    train()
