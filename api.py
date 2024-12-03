import joblib
import pandas as pd
import numpy as np
import re

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List


app = FastAPI()


# Вспомогательная функция конвертации еи в float
def convert_to_float(value):
  if isinstance(value, str):
    if value.strip().isalpha():  # Присваиваем пропуск, если значение невалидно
      return np.nan
    return float(value.split()[0])
  return value


# Вспомогательная функция конвертации torque
def extract_torque_values(value):
  if isinstance(value, str):
    # Общая предобработка строки
    value = value.lower()
    value = value.replace(',', '')

    # Другая единица измерения kgm
    if 'kgm' in value:
      torque_value_match = re.search(r'(\d+\.?\d*)\s*kgm', value)
      if torque_value_match:
        # Преобразование по формуле в nm
        torque_value = float(torque_value_match.group(1)) * 9.81
      else:
        torque_value = np.nan
    else:
        # Единица измерения nm
        torque_value_match = re.search(r'(\d+\.?\d*)\s*nm', value)
        if torque_value_match:
          torque_value = float(torque_value_match.group(1))
        else:
          # Число без единицы измерения
          torque_value_match = re.search(r'(\d+\.?\d*)', value)
          if torque_value_match:
              torque_value = float(torque_value_match.group(1))
          else:
              torque_value = np.nan

    # Парсинг max_torque_rpm
    max_rpm = np.nan

    # Если есть диапазон через '+/-'
    plus_minus_match = re.search(r'@ (\d+)[,]*\d*\s*\+/-\s*(\d+)', value)
    if plus_minus_match:
      # Правая граница
      max_rpm = int(plus_minus_match.group(1)) + int(plus_minus_match.group(2))
    else:
      # Если формат через слеш
      slash_match = re.search(r'(\d+\.?\d*)\s*/\s*(\d+)', value)
      if slash_match:
        # Значения до и после слэша
        torque_value = float(slash_match.group(1))
        max_rpm = int(slash_match.group(2))
      else:
        # Если обычный формат с диапазоном `-`
        rpm_match = re.search(r'@ (\d+)(?:-(\d+))?\s*(?:rpm)?', value)
        if rpm_match:
          # Правая граница
          max_rpm = int(rpm_match.group(2)) if rpm_match.group(2) else int(rpm_match.group(1))
        else:
          max_rpm = np.nan

    # Обработка строки с 'at' перед rpm
    at_rpm_match = re.search(r'(\d+\.?\d*)\s*(kgm|nm)?\s*at\s*(\d+)(?:-(\d+))?\s*rpm', value)
    if at_rpm_match:
      torque_value = float(at_rpm_match.group(1)) * 9.81 if at_rpm_match.group(2) == 'kgm' else float(at_rpm_match.group(1))
      max_rpm = int(at_rpm_match.group(3)) if at_rpm_match.group(4) is None else int(at_rpm_match.group(4))

    # Отдельная обработка для шаблонов типа 13.5@ 4,800(kgm@ rpm)
    kgm_rpm_match = re.search(r'(\d+\.?\d*)\s*@\s*(\d+)\s*\((kgm|nm)@\srpm\)', value)
    if kgm_rpm_match:
      torque_value = float(kgm_rpm_match.group(1)) * 9.81 if kgm_rpm_match.group(3) == 'kgm' else float(kgm_rpm_match.group(1))
      max_rpm = int(kgm_rpm_match.group(2))

    # Диапазоны rpm
    range_rpm_match = re.search(r'(\d+)-(\d+)\s*rpm', value)
    if range_rpm_match:
      # Правая граница
      max_rpm = int(range_rpm_match.group(2))

    return torque_value, max_rpm
    
  return np.nan, np.nan  # Если значение не строка


# Загружаем стандартизатор и модель
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
ridge_model = joblib.load("model.pkl")
medians = joblib.load("medians.pkl")

# Класс - признаки объекта
class Item(BaseModel):
    name: str
    year: int
    #selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


# Класс - признаки списка объектов
class Items(BaseModel):
    objects: List[Item]


# Доп функция преобразования датафрейма признаков объектов в нужный формат
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Разделение признаков для корректной стандартизации / кодировки OHE
    num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']
    cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']

    # Обрабатываем столбцы, как делали ранее на этапе EDA
    df['mileage'] = df['mileage'].apply(convert_to_float)
    df['engine'] = df['engine'].apply(convert_to_float)
    df['max_power'] = df['max_power'].apply(convert_to_float)

    df['seats'] = df['seats'].apply(int)
    df['engine'] = df['engine'].apply(int)

    df['name'] = df['name'].str.split(' ').str[0]
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(lambda x: pd.Series(extract_torque_values(x)))

    # Заполнение пропущенных значений
    for col in num_cols:
        if col in df.columns:
            print(col)
            print(medians)
            df[col].fillna(medians[col], inplace=True)
    
    # Стандартизация числовых признаков
    scaled_nums = scaler.transform(df[num_cols])

    # Кодирование категориальных признаков
    encoded_cat = encoder.transform(df[cat_cols])

    # Итоговый DataFrame
    processed_df = pd.concat(
        [
            pd.DataFrame(scaled_nums, columns=num_cols, index=df.index),
            pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        ],
        axis=1
    )

    return processed_df


# Метод принимает json представление признаков и выдает предсказанную стоимость
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])

    processed_df = preprocess_data(df)
    pred = ridge_model.predict(processed_df)[0]
        
    return round(pred, 2)


# Метод принимает список json представлений признаков и выдает предсказанные стоимости
@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    dfs = [pd.DataFrame([item.dict()]) for item in items.objects]
    df = pd.concat(dfs, ignore_index=True)

    processed_df = preprocess_data(df)
    preds = ridge_model.predict(processed_df)

    return preds.tolist()


# Метод принимает csv-файл с признаками объектов и выдает csv-файл с предсказаниями для них
@app.post("/predict_csv")
def predict_csv(file: UploadFile):
    df = pd.read_csv(file.file, sep=";")

    processed_df = preprocess_data(df)
    preds = ridge_model.predict(processed_df)

    # +1 колонка с предсказаниями
    df["predicted_selling_price"] = preds
    res_file = "file_with_preds.csv"
    df.to_csv(res_file, sep=";", index=False)

    return {"message": "Predictions added to csv", "file_path": res_file}