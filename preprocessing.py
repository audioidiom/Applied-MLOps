import pandas as pd
from sklearn import impute
# import re


def train_drop_duplicates(df_train):
    '''Удаляем дубликаты. Если признаковое описание одинаковое,
      а таргеты разные, оставляем первый встретившийся объект'''

    df_no_doubles = df_train.drop_duplicates(keep='first')

    idx_first_obj = df_no_doubles.drop(columns='selling_price').drop_duplicates(keep='first').index
    df_train = df_no_doubles[pd.Series(dict((x, x in idx_first_obj) for x in df_no_doubles.index))].copy()

    df_train = df_train.reset_index(drop=True)
    
    return df_train


def mileage_clean_n_cast(df_train, df_test):
    ''' Удаляем из признака единицы измерения и приводим типы'''

    df_train['mileage'] = df_train['mileage'].map(lambda x:
                                                  float(str(x)
                                                        .rstrip(' kmpl')
                                                        .rstrip(' km/kg')
                                                        )
                                                  )
    df_test['mileage'] = df_test['mileage'].map(lambda x:
                                                float(str(x)
                                                      .rstrip(' kmpl')
                                                      .rstrip(' km/kg')
                                                      )
                                                )
    return df_train, df_test


def engine_clean_n_cast(df_train, df_test):
    ''' Удаляем из признака единицы измерения и приводим типы'''

    df_train['engine'] = df_train['engine'].map(lambda x:
                                                float(str(x)
                                                      .rstrip(' CC')
                                                      )
                                                )
    df_test['engine'] = df_test['engine'].map(lambda x:
                                              float(str(x)
                                                    .rstrip(' CC')
                                                    )
                                              )
    return df_train, df_test


def max_power_clean_n_cast(df_train, df_test):
    ''' Удаляем из признака единицы измерения и приводим типы'''

    df_train['max_power'] = df_train['max_power'].map(lambda x:
                                                      str(x)
                                                      .rstrip(' bhp')
                                                      .rstrip('bhp')
                                                      )
    df_test['max_power'] = df_test['max_power'].map(lambda x:
                                                    str(x)
                                                    .rstrip(' bhp')
                                                    .rstrip('bhp')
                                                    )

    # Заменим '' на nan
    df_train['max_power'] = df_train['max_power'].replace(to_replace='^$',
                                                          value='nan',
                                                          regex=True)

    df_train['max_power'] = df_train['max_power'].astype('float64')
    df_test['max_power'] = df_test['max_power'].astype('float64')
    return df_train, df_test


def filling_na_with_medians(df_train, df_test):
    ''' Заполняем пустые значения столбцов в train и test медианами
        этих столбцов из train'''

    mask_train = df_train.isna().any()
    mask_test = df_test.isna().any()

    X_notna_train = df_train[df_train.columns[~mask_train].values]
    X_notna_test = df_test[df_test.columns[~mask_test].values]

    X_nums_na_train = df_train[df_train.columns[mask_train].values]
    X_nums_na_test = df_test[df_test.columns[mask_test].values]

    na_replacer = impute.SimpleImputer(strategy='median')
    na_replacer.fit(X_nums_na_train)

    X_filled_train = pd.DataFrame(data=na_replacer.transform(X_nums_na_train),
                                  columns=df_train.columns[mask_train]
                                                  .values)
    X_filled_test = pd.DataFrame(data=na_replacer.transform(X_nums_na_test),
                                 columns=df_test.columns[mask_test]
                                                .values)

    df_train = pd.concat([X_notna_train, X_filled_train], axis=1)
    print(df_train.isna().any())
    df_test = pd.concat([X_notna_test, X_filled_test], axis=1)

    return df_train, df_test