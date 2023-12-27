# TODO: логгирование для инфера
import random

import hydra
import numpy as np
import pandas as pd
import skops.io as skops
from dvc.api import DVCFileSystem

import preprocessing as pp


random.seed(42)
np.random.seed(42)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def infer(cfg):
    # Загружаем данные из GDrive
    fs = DVCFileSystem(cfg["source_repo"], cfg["rev"])
    fs.get("data", cfg["data_path"], recursive=True)
    df_test = pd.read_csv(cfg["infer"]["df_path"])

    print("Test data shape: ", df_test.shape)

    # Очищаем столбцы от единиц измерения и приводим к float
    df_test = pp.mileage_clean_n_cast(df_test)
    df_test = pp.engine_clean_n_cast(df_test)
    df_test = pp.max_power_clean_n_cast(df_test)

    # Столбец torque просто дропнем
    df_test = df_test.drop(columns="torque")

    # Заполняем пустые значения медианами
    df_test = pp.filling_na_with_medians(cfg, df_test, cfg["infer"]["mode"])

    # Кастуем к инту
    df_test["engine"] = df_test["engine"].astype("int64")
    df_test["seats"] = df_test["seats"].astype("int64")

    # Дроп ненужных столбцов
    if "selling_price" in df_test.columns:
        X_test = df_test.drop(columns=["selling_price", "name"])
    else:
        X_test = df_test.drop(columns=["name"])

    # стандартизируем вещественные признаки
    X_test_num_scaled = pp.normalize_numerical(
        cfg, X_test, cfg["infer"]["mode"]
    )

    # кодируем категориальные признаки
    X_test_cat_coded = pp.encoding_categorical(
        cfg, X_test, cfg["infer"]["mode"]
    )

    # объединяем стандартизованный и закодированный элементы
    X_test_preprocessed = X_test_num_scaled.merge(
        X_test_cat_coded, how="inner", left_index=True, right_index=True
    )
    X_test_preprocessed = X_test_preprocessed.to_numpy()

    # Импорт модели
    ridge_reg = skops.load(cfg["model_path"])

    # Получаем предсказания модели и сохраняем их
    y_pred = ridge_reg.predict(X_test_preprocessed)
    pd.Series(y_pred).to_csv(cfg["infer"]["pred_csv_path"])


if __name__ == "__main__":
    infer()
