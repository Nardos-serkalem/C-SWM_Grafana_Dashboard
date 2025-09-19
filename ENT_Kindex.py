import os
import pandas as pd
import numpy as np
import ftplib
import logging
import time
import tempfile
from datetime import datetime, timedelta, timezone
from scipy.signal import medfilt
from scipy.stats import zscore
from prometheus_client import Gauge, start_http_server

len_days = 3
update_interval_minutes = 10  
k9_limit = 500 
ftp_server = "ftp.gfz-potsdam.de"
ftp_path = "/pub/home/obs/data/iaga2002/ENT0/"
station_code = "ent"
temp_dir = tempfile.gettempdir()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

k_index_gauge = Gauge("geomagnetic_k_index", "K-index value", ["station"])

def list_ftp_files(ftp, path):
    files = []
    ftp.cwd(path)
    ftp.dir(files.append)
    parsed_files = []
    for line in files:
        parts = line.split()
        if len(parts) < 9:
            continue
        filename = parts[-1]
        if filename.lower().startswith(station_code.lower()) and filename.lower().endswith('pmin.min'):
            try:
                date_str = filename[len(station_code):len(station_code)+8]
                file_date = datetime.strptime(date_str, "%Y%m%d")
                parsed_files.append((file_date, filename))
            except ValueError:
                continue
    parsed_files.sort(key=lambda x: x[0], reverse=True)
    return [f[1] for f in parsed_files[:len_days]]

def get_ftp_files():
    try:
        from ftplib import FTP
        ftp = FTP(ftp_server, timeout=60)
        ftp.login()
        try:
            files_to_download = list_ftp_files(ftp, ftp_path)
            if not files_to_download:
                logging.error(f"No matching files found in {ftp_path}")
                return []
            local_files = []
            for filename in files_to_download:
                local_path = os.path.join(temp_dir, filename)
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)
                local_files.append(local_path)
                logging.info(f"Downloaded {filename} to {local_path}")
            return local_files
        finally:
            ftp.quit()
    except Exception as e:
        logging.error(f"FTP connection error: {e}")
        return []

def read_iaga2002(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        header_end = -1
        for i, line in enumerate(lines):
            if line.startswith("DATE") and "TIME" in line and "DOY" in line:
                header_end = i
                break
        if header_end == -1:
            logging.error(f"Header not found in file: {file_path}")
            return pd.DataFrame(), None, None

        header_fields = [field.strip().replace('|', '') for field in lines[header_end].split()]
        reported_components = ['X', 'Y', 'Z'] if all(c in header_fields for c in ['X', 'Y', 'Z']) else None
        if reported_components is None:
            logging.error(f"No valid components in file {file_path}")
            return pd.DataFrame(), None, None

        data = pd.read_csv(
            file_path,
            skiprows=header_end + 1,
            names=header_fields,
            sep=r"\s+",
            na_values=[99999.00, 99999.9],
            engine='python'
        )
        datetime_col = data["DATE"].astype(str) + " " + data["TIME"].astype(str)
        data["DATETIME"] = pd.to_datetime(datetime_col, errors='coerce', utc=True)
        data.dropna(subset=["DATETIME"], inplace=True)
        for comp in reported_components:
            data[comp] = pd.to_numeric(data[comp], errors='coerce')
        return data[["DATETIME"] + reported_components], reported_components, station_code.upper()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return pd.DataFrame(), None, None

def preprocess_data(data, components):
    if data.empty:
        return data
    numeric_data = data[components].select_dtypes(include=np.number)
    if not numeric_data.empty:
        z_scores = numeric_data.apply(lambda x: zscore(x, nan_policy='omit') if x.std(skipna=True) > 0 else np.zeros_like(x))
        valid_mask = (np.abs(z_scores) <= 2.5).all(axis=1)
        return data[valid_mask]
    return data

def time_to_float(dt_series):
    if dt_series.empty:
        return np.array([])
    return (dt_series.dt.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')

def calculate_k_index(minute_time_float, minute_comp_x, minute_comp_y, k9):
    if len(minute_time_float) == 0:
        return np.array([]), np.array([])
    first_dt_utc = datetime.fromtimestamp(minute_time_float[0], tz=timezone.utc)
    start_of_first_day_utc = datetime(first_dt_utc.year, first_dt_utc.month, first_dt_utc.day, tzinfo=timezone.utc)
    day_seconds_start_utc = (start_of_first_day_utc - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
    hour_blocks = (minute_time_float - day_seconds_start_utc) // 10800
    variations, timestamps_float = [], []
    for block_idx in np.unique(hour_blocks):
        mask = hour_blocks == block_idx
        variation = max(np.ptp(minute_comp_x[mask]), np.ptp(minute_comp_y[mask])) if np.sum(mask) > 1 else 0
        variations.append(variation)
        timestamps_float.append(day_seconds_start_utc + block_idx * 10800 + 5400)
    thresholds = np.array([0, 5, 10, 20, 40, 70, 120, 200, 330, 500]) * k9 / 500.0
    k_values_indices = np.searchsorted(thresholds, variations, side='right') - 1
    k_values = np.clip(k_values_indices, 0, 9).astype(float)
    k_values[k_values == 0] = 0.25
    return np.array(k_values), np.array(timestamps_float)

# promethus expose
def expose_k_index(k_values, station_name="ENT"):
    if len(k_values) > 0:
        k_index_gauge.labels(station=station_name).set(float(k_values[-1]))

def main_loop():
    while True:
        try:
            logging.info("Fetching data...")
            files = get_ftp_files()
            if not files:
                time.sleep(update_interval_minutes * 60)
                continue

            all_data, components, station_name = pd.DataFrame(), None, station_code.upper()
            for file_path in files:
                data, comps, name = read_iaga2002(file_path)
                if not data.empty:
                    all_data = pd.concat([all_data, data], ignore_index=True)
                    if components is None:
                        components, station_name = comps, name

            if all_data.empty:
                time.sleep(update_interval_minutes * 60)
                continue

            all_data = preprocess_data(all_data, components)
            times_float = time_to_float(all_data["DATETIME"])
            comp_x, comp_y = all_data[components[0]].values, all_data[components[1]].values
            k_indices, k_times = calculate_k_index(times_float, comp_x, comp_y, k9_limit)

            expose_k_index(k_indices, station_name)
            logging.info(f"K-index exposed for {station_name}: {k_indices[-1] if len(k_indices) else 'N/A'}")

            time.sleep(update_interval_minutes * 60)
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    start_http_server(8000)
    logging.info("Prometheus exporter started at http://localhost:8000/metrics")
    main_loop()
