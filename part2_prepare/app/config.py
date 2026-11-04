import os
from dotenv import load_dotenv

# Загружаем переменные из файла .env (если он есть)
load_dotenv()

# Настройки по умолчанию (если в .env ничего не указано)
DEFAULT_INPUT_PATH = os.getenv("DATA_PATH", "../data/input")
DEFAULT_OUTPUT_PATH = os.getenv("OUTPUT_PATH", "../data/output/dataset.npy")