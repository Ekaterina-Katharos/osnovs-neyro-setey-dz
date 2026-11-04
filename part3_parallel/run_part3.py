import sys
import time
import logging
import multiprocessing
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR / "app"
sys.path.append(str(APP_DIR))

from producer_consumer import run_parallel  # noqa: E402
from performance import sequential_process, save_report  # noqa: E402
from logging_config import setup_logging  # noqa: E402


def main():
    """Main function to run parallel data processing."""
    setup_logging()
    input_dir = "data/input"
    ext = ['.jpg', '.png', '.jpeg']
    path_obj = Path(input_dir)
    file_paths = [
        f for f in path_obj.rglob('*')
        if f.is_file() and f.suffix.lower() in ext
    ]
    if not file_paths:
        logging.error("Files not found! Check data/input folder.")
        return

    logging.info(f"Found files: {len(file_paths)}")

    logging.info("Starting SEQUENTIAL processing...")
    seq_time, seq_count = sequential_process(file_paths)
    logging.info(f"Sequential processing finished in {seq_time:.4f} sec.")

    logging.info("Starting PARALLEL processing...")
    start_par = time.time()
    par_count = run_parallel(input_dir, ext, num_threads=4)
    par_time = time.time() - start_par
    logging.info(f"Parallel processing finished in {par_time:.4f} sec.")

    save_report(seq_time, par_time, par_count)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
