import sys
import os
import time
import logging
import multiprocessing
from pathlib import Path

# Add app folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from producer_consumer import run_parallel
from performance import sequential_process, save_report
from logging_config import setup_logging

def main():
    # Setup logging
    setup_logging()
    
    input_dir = "../data/input"
    ext = ['.jpg', '.png', '.jpeg']
    
    # Collect files
    path_obj = Path(input_dir)
    file_paths = [f for f in path_obj.rglob('*') if f.is_file() and f.suffix.lower() in ext]
    
    if not file_paths:
        logging.error("Files not found! Check data/input.")
        return

    logging.info(f"Found files: {len(file_paths)}")

    # 1. Sequential processing
    logging.info("Starting SEQUENTIAL processing...")
    seq_time, seq_count = sequential_process(file_paths)
    logging.info(f"Sequential processing finished in {seq_time:.4f} sec.")

    # 2. Parallel processing
    logging.info("Starting PARALLEL processing...")
    start_par = time.time()
    par_count = run_parallel(input_dir, ext, num_threads=4)
    par_time = time.time() - start_par
    logging.info(f"Parallel processing finished in {par_time:.4f} sec.")

    # 3. Save report
    save_report(seq_time, par_time, par_count)

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()