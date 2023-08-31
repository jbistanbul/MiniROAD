import logging
import os

def get_logger(output_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) 
    # formatter = logging.Formatter('%(asctime)s:%(message)s ', '%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    log_file = os.path.join(output_path, 'log.txt')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger