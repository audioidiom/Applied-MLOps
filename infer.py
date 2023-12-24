import random

import numpy as np
import pandas as pd
import skops.io as skops
from dvc.api import DVCFileSystem

import preprocessing as pp


random.seed(42)
np.random.seed(42)


def infer():
    # Загружаем данные из GDrive
    url_repo = "git@github.com:audioidiom/Applied-MLOps.git"
    rev = "master"
    fs = DVCFileSystem(url_repo, rev)
    path = "data"
    fs.get("data", path, recursive=True)

    test_path = path + "/" + "datasets/" + "cars_test.csv"
    df_test = pd.read_csv(test_path)

    print("Test data shape: ", df_test.shape)

    # Очищаем столбцы от единиц измерения и приводим к float
    df_test = pp.mileage_clean_n_cast(df_test)
    df_test = pp.engine_clean_n_cast(df_test)
    df_test = pp.max_power_clean_n_cast(df_test)

    # Столбец torque просто дропнем
    df_test = df_test.drop(columns="torque")

    # Заполняем пустые значения медианами
    df_test = pp.filling_na_with_medians(df_test, "infer")

    # Кастуем к инту
    df_test["engine"] = df_test["engine"].astype("int64")
    df_test["seats"] = df_test["seats"].astype("int64")

    # Дроп ненужных столбцов
    if "selling_price" in df_test.columns:
        X_test = df_test.drop(columns=["selling_price", "name"])
    else:
        X_test = df_test.drop(columns=["name"])

    # стандартизируем вещественные признаки
    X_test_num_scaled = pp.normalize_numerical(X_test, "infer")

    # кодируем категориальные признаки
    X_test_cat_coded = pp.encoding_categorical(X_test, "infer")

    # объединяем стандартизованный и закодированный элементы
    X_test_preprocessed = X_test_num_scaled.merge(
        X_test_cat_coded, how="inner", left_index=True, right_index=True
    )
    X_test_preprocessed = X_test_preprocessed.to_numpy()

    # Импорт модели
    # ridge_reg = skops.load("data/main_model/ridge.skops")
    ridge_reg = skops.load("data/main_model/ridge.skops", trusted=True)

    # Получаем предсказания модели и сохраняем их
    y_pred = ridge_reg.predict(X_test_preprocessed)
    pd.Series(y_pred).to_csv("data/predictions/y_pred.csv")


if __name__ == "__main__":
    infer()
