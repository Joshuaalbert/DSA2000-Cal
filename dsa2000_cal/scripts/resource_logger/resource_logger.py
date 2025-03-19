#!/usr/bin/env python3
import argparse
import datetime
import logging
import socket
import time

import psutil

# Try importing pynvml to get GPU metrics.
try:
    import pynvml

    pynvml.nvmlInit()
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
except ImportError:
    GPU_COUNT = 0
    print("pynvml not installed. GPU metrics will not be available.")


def setup_logger(log_file):
    """
    Set up the logger with the required format.
    """
    logger = logging.getLogger("ResourceLogger")
    logger.setLevel(logging.INFO)

    # Create a file handler that logs to the specified file.
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s  %(node)s [%(resource_type)s]: %(value)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_resource(logger, node, resource_type, value):
    """
    Log a single resource metric using the given logger.
    """
    logger.info("", extra={'node': node, 'resource_type': resource_type, 'value': value})


def log_resources(logger, node):
    """
    Gather system resource metrics and log them.
    """
    # Log RAM usage (used memory in GB)
    mem = psutil.virtual_memory()
    mem_usage_gb = mem.used / (1024 ** 3)
    log_resource(logger, node, "MEMORY", f"{mem_usage_gb:.2f} GB")

    # Log CPU usage percentage (blocking call to measure over a short interval)
    cpu_usage = psutil.cpu_percent(interval=1)
    log_resource(logger, node, "CPU", f"{cpu_usage:.1f}%")

    # Log CPU temperature if available.
    try:
        temps = psutil.sensors_temperatures()
        # Try common keys; if not, use the first available sensor.
        cpu_temp = None
        for sensor in ("coretemp", "cpu_thermal"):
            if sensor in temps and temps[sensor]:
                cpu_temp = temps[sensor][0].current
                break
        if cpu_temp is None:
            # Fall back to any available sensor if the common ones aren't present.
            for sensor_entries in temps.values():
                if sensor_entries:
                    cpu_temp = sensor_entries[0].current
                    break
        if cpu_temp is not None:
            log_resource(logger, node, "CPU_TEMPERATURE", f"{cpu_temp:.1f}°C")
        else:
            log_resource(logger, node, "CPU_TEMPERATURE", "N/A")
    except Exception as e:
        # In case sensors_temperatures is not supported.
        log_resource(logger, node, "CPU_TEMPERATURE", "N/A")

    # Log GPU metrics if available
    if GPU_COUNT > 0:
        for i in range(GPU_COUNT):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # GPU memory usage in GB
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_usage_gb = mem_info.used / (1024 ** 3)
            log_resource(logger, node, f"GPU_MEMORY[{i}]", f"{gpu_mem_usage_gb:.2f} GB")

            # GPU utilisation percentage
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            log_resource(logger, node, f"GPU[{i}]", f"{util.gpu}%")

            # GPU temperature (in °C)
            gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            log_resource(logger, node, f"GPU_TEMPERATURE[{i}]", f"{gpu_temp}°C")


def main():
    parser = argparse.ArgumentParser(description="Log resource usage to a file at a given cadence.")
    parser.add_argument("--interval", type=int, default=5,
                        help="Logging interval in seconds")
    parser.add_argument("--duration", type=int, default=1000,
                        help="Logging interval in hours")
    parser.add_argument("--logfile", type=str, default="resource_usage.log",
                        help="Log file path")
    args = parser.parse_args()

    # Convert the provided interval to a timedelta object.
    cadence = datetime.timedelta(seconds=args.interval)
    duration = datetime.timedelta(hours=args.duration)

    node = socket.gethostname()
    logger = setup_logger(args.logfile)

    # Infinite loop to log resource usage every `cadence` seconds.
    t0 = time.time()
    while True:
        log_resources(logger, node)
        # Sleep for the duration of the cadence.
        time.sleep(cadence.total_seconds())

        # Break if the duration has elapsed.
        if time.time() - t0 > duration.total_seconds():
            break


if __name__ == '__main__':
    main()
