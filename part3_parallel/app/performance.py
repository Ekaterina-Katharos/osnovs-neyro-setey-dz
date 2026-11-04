import time
import json
import os
import numpy as np
from PIL import Image
from augment import apply_augmentation


def sequential_process(file_paths):
    """Последовательная обработка (для сравнения)."""
    start_time = time.time()
    result = []
    for path in file_paths:
        try:
            with Image.open(path) as img:
                img_array = np.array(img) / 255.0
                augmented = apply_augmentation(img_array)
                result.append(augmented)
        except Exception as e:
            print(f"Ошибка {path}: {e}")
    elapsed = time.time() - start_time
    return elapsed, len(result)


def save_report(seq_time, par_time, count):
    """Сохраняет отчет в JSON."""
    report = {
        "files_processed": count,
        "sequential_time_sec": round(seq_time, 4),
        "parallel_time_sec": round(par_time, 4),
        "speedup": round(seq_time / par_time, 2) if par_time > 0 else 0,
    }
    os.makedirs("../data/output", exist_ok=True)
    with open("../data/output/performance_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print("Отчет сохранен в performance_report.json")
    print(json.dumps(report, indent=4, ensure_ascii=False))
