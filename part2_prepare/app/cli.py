import argparse
import sys
import subprocess
import importlib.util
from scanner import prepare_data
from config import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH

def run_doctor():
    print("--- ДИАГНОСТИКА ОКРУЖЕНИЯ ---")
    print(f"Версия Python: {sys.version.split()[0]}")
    
    libs = ['numpy', 'PIL', 'pandas']
    for lib in libs:
        if importlib.util.find_spec(lib):
            print(f"[OK] Библиотека {lib} установлена")
        else:
            print(f"[ВНИМАНИЕ] Библиотека {lib} НЕ найдена")
            
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] CUDA доступна (nvidia-smi работает)")
        else:
            print("[ИНФО] CUDA не найдена")
    except FileNotFoundError:
        print("[ИНФО] Утилита nvidia-smi не найдена (нет видеокарты Nvidia)")

    print(f"Текущая рабочая директория: {sys.path[0]}")

def main():
    parser = argparse.ArgumentParser(description="Утилита подготовки данных")
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # Команда prepare
    p_prepare = subparsers.add_parser("prepare", help="Подготовка данных из папки в .npy")
    p_prepare.add_argument("--path", default=DEFAULT_INPUT_PATH, help="Путь к папке с данными")
    p_prepare.add_argument("--ext", nargs='+', default=['.jpg', '.png'], help="Список расширений")
    p_prepare.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Путь для сохранения .npy")

    # Команда doctor
    subparsers.add_parser("doctor", help="Проверка окружения")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_data(args.path, args.ext, args.output)
    elif args.command == "doctor":
        run_doctor()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()