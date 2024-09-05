import psutil
import time
def get_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_info = psutil.virtual_memory()
    ram_percent = ram_info.percent
    start_time = time.time()
    return cpu_percent, ram_percent, start_time