import threading
import queue
import multiprocessing
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from augment import apply_augmentation

# Глобальный счетчик и лок для потокобезопасности
counter = 0
lock = threading.Lock()


def producer(directory, ext, file_queue):
    """Поток-производитель: ищет файлы и кладет их пути в очередь."""
    logging.info("Производитель начал работу.")
    path_obj = Path(directory)
    files = [f for f in path_obj.rglob("*") if f.is_file() and f.suffix.lower() in ext]

    for f in files:
        file_queue.put(f)

    # Отправляем сигналы завершения (None) для каждого потока-потребителя
    for _ in range(4):
        file_queue.put(None)
    logging.info(f"Производитель нашел {len(files)} файлов и завершил работу.")


def consumer(file_queue, array_queue):
    """Поток-потребитель: читает файлы и кладет массивы в другую очередь."""
    global counter
    while True:
        file_path = file_queue.get()
        if file_path is None:
            file_queue.task_done()
            break
        try:
            with Image.open(file_path) as img:
                img_array = np.array(img) / 255.0
                array_queue.put(img_array)
                with lock:
                    counter += 1
        except Exception as e:
            logging.error(f"Ошибка чтения {file_path}: {e}")
        finally:
            file_queue.task_done()


def run_parallel(input_dir, ext, num_threads=4):
    """Запускает многопоточное чтение и многопроцессорную аугментацию."""
    global counter
    counter = 0

    file_queue = queue.Queue()
    array_queue = queue.Queue()

    # Запускаем поток-производитель
    prod_thread = threading.Thread(target=producer, args=(input_dir, ext, file_queue))

    # Запускаем потоки-потребители
    consumers = []
    for _ in range(num_threads):
        t = threading.Thread(target=consumer, args=(file_queue, array_queue))
        consumers.append(t)
        t.start()

    prod_thread.start()

    # Пул процессов для аугментации
    with multiprocessing.Pool() as pool:
        results = []
        # Достаем массивы из очереди и отправляем в процессы
        while any(t.is_alive() for t in consumers) or not array_queue.empty():
            try:
                arr = array_queue.get(timeout=0.1)
                # apply_async позволяет не блокировать основной поток
                res = pool.apply_async(apply_augmentation, (arr,))
                results.append(res)
            except queue.Empty:
                continue

        # Ждем завершения потоков и собираем результаты
        for t in consumers:
            t.join()

        augmented_data = [res.get() for res in results]

    logging.info(f"Обработано файлов: {counter}")
    return len(augmented_data)
