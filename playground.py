import pandas as pd

data = {
    'City_ID': [1, 2, 3],
    'X_Coordinate': [565.0, 25.0, 345.0],
    'Y_Coordinate': [575.0, 185.0, 750.0]
}

df = pd.DataFrame(data)
print(df)

subset = df.iloc[0:2, 1:3]  # İlk iki satır, 2. ve 3. sütun
print(subset)