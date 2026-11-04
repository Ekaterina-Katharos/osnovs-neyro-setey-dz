import logging
import os


def setup_logging(log_level_str="INFO"):
    """
    Настраивает логирование для приложения.
    """
    # Получаем числовой уровень логирования (по умолчанию INFO)
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Создаем папку для логов, если ее нет (в корне проекта)
    log_dir = "../logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Настройка формата сообщений
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Настраиваем базовую конфигурацию
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "app.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)
