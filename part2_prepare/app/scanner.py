import os
import time
import numpy as np
from pathlib import Path
from PIL import Image


def prepare_data(input_path, extensions, output_file):
    """
    Сканирует директорию, читает файлы, нормализует и сохраняет в .npy.
    """
    print(f"Сканирование директории: {input_path}")
    start_time = time.time()

    extensions = [
        ext.lower() if ext.startswith(".") else "." + ext.lower() for ext in extensions
    ]

    path_obj = Path(input_path)
    if not path_obj.exists():
        print(f"Ошибка: Директория {input_path} не найдена.")
        return

    # Рекурсивно ищем все файлы с нужными расширениями
    files = [
        f for f in path_obj.rglob("*") if f.is_file() and f.suffix.lower() in extensions
    ]

    if not files:
        print("Файлы с указанными расширениями не найдены.")
        return

    data_list = []
    print(f"Найдено файлов: {len(files)}. Начинаем обработку...")

    for file in files:
        try:
            if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                with Image.open(file) as img:
                    img_array = np.array(img)
                    # Нормализация: делим на 255 (для 8-битных изображений)
                    normalized_array = img_array / 255.0
                    data_list.append(normalized_array)
            elif file.suffix.lower() == ".csv":
                data_list.append(np.genfromtxt(file, delimiter=","))
        except Exception as e:
            print(f"Ошибка чтения файла {file}: {e}")

    if data_list:
        try:
            # Склеиваем все массивы в один
            final_dataset = np.stack(data_list)
            # Создаем папку для вывода, если ее нет
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.save(output_file, final_dataset)

            elapsed_time = time.time() - start_time
            print("\n--- Статистика ---")
            print(f"Успешно обработано файлов: {len(data_list)}")
            print(f"Размер итогового массива: {final_dataset.shape}")
            print(f"Время обработки: {elapsed_time:.2f} секунд")
            print(f"Данные сохранены в: {output_file}")
        except ValueError as e:
            print(
                f"Ошибка объединения массивов (возможно, разные размеры картинок): {e}"
            )
