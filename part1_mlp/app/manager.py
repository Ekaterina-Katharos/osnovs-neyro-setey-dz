import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetManager:
    """Класс для загрузки и подготовки данных."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def prepare_data(self, filepath: str, test_size=0.2):
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {filepath} не найден.")
        except Exception as e:
            raise ValueError(f"Ошибка чтения: {e}")

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)

        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        print(
            f"Данные загружены. Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}"
        )
