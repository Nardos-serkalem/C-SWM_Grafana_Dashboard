import os
import pandas as pd
import numpy as np
import ftplib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from scipy.signal import medfilt
import logging
import time
import tempfile
from scipy.stats import zscore
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

plt.rcParams.update({'font.size': 14})
plt.style.use('dark_background')

len_days = 3
update_interval_minutes = 10  
k9_limit = 500 

ftp_server = "ftp.gfz-potsdam.de"
ftp_path = "/pub/home/obs/data/iaga2002/ENT0/"
station_code = "ent"
temp_dir = tempfile.gettempdir()


influx_url = "http://localhost:8086"
influx_token = "jY7_rv4hnA4r79A1LljVFqa6tbeZuzYFgaSe7tu7qZBbwegfo8P9KaJBaF9UZ5GbKLRu4YLDBF96Sh2KI0XyHQ=="  
influx_org = "nardos_org"      
influx_bucket = "space_weather"  


client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
write_api = client.write_api(write_options=SYNCHRONOUS)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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

        # Find the header line
        header_end = -1
        for i, line in enumerate(lines):
            if line.startswith("DATE") and "TIME" in line and "DOY" in line:
                header_end = i
                break

        if header_end == -1:
            logging.error(f"Header not found in file: {file_path}")
            return pd.DataFrame(), None, None

        header_fields = [field.strip().replace('|', '') for field in lines[header_end].split()]

        reported_components = None
        station_name = f"{station_code.upper()} (Entoto)"

        # Determine which components exist
        possible_xyz = [f'{station_code.upper()}X', f'{station_code.upper()}Y', f'{station_code.upper()}Z']
        possible_hdz = [f'{station_code.upper()}H', f'{station_code.upper()}D', f'{station_code.upper()}Z']

        if all(c in header_fields for c in ['X', 'Y', 'Z']):
            reported_components = ['X', 'Y', 'Z']
        elif all(c in header_fields for c in ['H', 'D', 'Z']):
            reported_components = ['H', 'D', 'Z']
        elif all(c in header_fields for c in possible_xyz):
            reported_components = ['X', 'Y', 'Z']
        elif all(c in header_fields for c in possible_hdz):
            reported_components = ['H', 'D', 'Z']
        else:
            # Try to read from metadata lines
            for line in lines[:header_end]:
                if line.startswith("Reported"):
                    reported_str = ''.join(filter(str.isalpha, line.split()[-1])).upper()
                    if reported_str in ["XYZF", "XYZ"]:
                        reported_components = ['X', 'Y', 'Z']
                    elif reported_str in ["HDZF", "HDZ"]:
                        reported_components = ['H', 'D', 'Z']
                elif line.startswith("Station Name"):
                    try:
                        station_name = line.split(':', 1)[1].split('|')[0].strip()
                    except:
                        pass

        if reported_components is None:
            logging.error(f"No valid magnetic components found in file: {file_path}")
            return pd.DataFrame(), None, None

        # Read data
        data = pd.read_csv(
            file_path,
            skiprows=header_end + 1,
            names=header_fields,
            sep='\s+',
            na_values=[99999.00, 99999.9, '99999.00', '99999.9'],
            engine='python'
        )

        # Rename columns if needed
        rename_dict = {}
        if f'{station_code.upper()}X' in data.columns:
            rename_dict.update({
                f'{station_code.upper()}X': 'X',
                f'{station_code.upper()}Y': 'Y',
                f'{station_code.upper()}Z': 'Z'
            })
        elif f'{station_code.upper()}H' in data.columns:
            rename_dict.update({
                f'{station_code.upper()}H': 'H',
                f'{station_code.upper()}D': 'D',
                f'{station_code.upper()}Z': 'Z'
            })
        if rename_dict:
            data.rename(columns=rename_dict, inplace=True)

        # Parse datetime
        try:
            datetime_col = data["DATE"].astype(str) + " " + data["TIME"].astype(str)
            data["DATETIME"] = pd.to_datetime(datetime_col, errors='coerce', utc=True)
            data.dropna(subset=["DATETIME"], inplace=True)
        except Exception as e:
            logging.error(f"Error parsing datetime in file {file_path}: {e}")
            return pd.DataFrame(), None, None

        # Ensure all components are numeric
        for comp in reported_components:
            if comp in data.columns:
                data[comp] = pd.to_numeric(data[comp], errors='coerce')
            else:
                logging.error(f"Component {comp} missing in file {file_path}")
                return pd.DataFrame(), None, None

        return data[["DATETIME"] + reported_components], reported_components, station_name

    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return pd.DataFrame(), None, None

def preprocess_data(data, components):
    if data.empty:
        return data
    numeric_data = data[components].select_dtypes(include=np.number)
    if not numeric_data.empty:
        z_scores = numeric_data.apply(lambda x: zscore(x, nan_policy='omit') if x.std(skipna=True) > 0 else np.zeros_like(x))
        z_scores = np.abs(z_scores)
        valid_mask = (z_scores <= 2.5).all(axis=1)
        filtered_data = data[valid_mask]
        return filtered_data
    return data

def time_to_float(dt_series):
    if dt_series.empty:
        return np.array([])
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize("UTC")
    return (dt_series - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')

def calculate_k_index(minute_time_float, minute_comp_x, minute_comp_y, k9):
    if len(minute_time_float) == 0:
        return np.array([]), np.array([])
    first_dt_utc = datetime.fromtimestamp(minute_time_float[0], tz=timezone.utc)
    start_of_first_day_utc = datetime(first_dt_utc.year, first_dt_utc.month, first_dt_utc.day, tzinfo=timezone.utc)
    day_seconds_start_utc = (start_of_first_day_utc - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
    hour_blocks = (minute_time_float - day_seconds_start_utc) // 10800
    variations = []
    timestamps_float = []
    unique_blocks = np.unique(hour_blocks)

    for block_idx in unique_blocks:
        mask = hour_blocks == block_idx
        if np.sum(mask) > 1:
            range_x = np.ptp(minute_comp_x[mask])
            range_y = np.ptp(minute_comp_y[mask])
            variation = max(range_x, range_y)
        else:
            variation = 0
        variations.append(variation)
        timestamps_float.append(day_seconds_start_utc + block_idx * 10800 + 5400)

    niemegk_thresholds = np.array([0, 5, 10, 20, 40, 70, 120, 200, 330, 500])
    scaled_thresholds = niemegk_thresholds * k9 / 500.0
    k_values_indices = np.searchsorted(scaled_thresholds, variations, side='right') - 1
    k_values_indices = np.clip(k_values_indices, 0, 9)
    k_values = k_values_indices.astype(float)
    k_values[k_values == 0] = 0.25
    return np.array(k_values), np.array(timestamps_float)

def compute_derivatives(data):
    data = data.sort_values("DATETIME").reset_index(drop=True)
    if 'X' in data.columns and 'Y' in data.columns and 'H' not in data.columns:
        data["H"] = np.sqrt(data['X']**2 + data['Y']**2)
    if 'D' in data.columns:
        data["D_deg"] = data["D"] / 60.0
        data["X"] = data["H"] * np.cos(np.radians(data["D_deg"]))
    if 'X' in data.columns:
        data["dX/dt"] = data["X"].diff().abs().fillna(0)
        data["dX/dt_smooth"] = medfilt(data["dX/dt"], kernel_size=5)
    if 'F' in data.columns:
        data["dF/dt"] = data["F"].diff().abs().fillna(0)
        data["dF/dt_smooth"] = medfilt(data["dF/dt"], kernel_size=5)
    if 'H' in data.columns:
        data["dH/dt"] = data["H"].diff().abs().fillna(0)
        data["dH/dt_smooth"] = medfilt(data["dH/dt"], kernel_size=5)
    return data

def plot_k_indices_with_derivatives(data, k_indices, k_times, station_name=""):
    if data.empty or len(k_indices) == 0:
        return
    k_times_dt = pd.to_datetime(k_times, unit='s', utc=True)
    start_time = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=len_days)
    recent_data = data[data["DATETIME"] >= start_time].copy()
    if recent_data.empty:
        return
    if 'dX/dt_smooth' not in recent_data.columns:
        recent_data = compute_derivatives(recent_data)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"{station_name} Magnetic Indices and Derivatives\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    colors = ["green" if k < 5 else "yellow" if k == 5 else "#DAA520" if k == 6 else "orange" if k == 7 else "red" if k == 8 else "darkred" for k in k_indices]
    for i, (time, k) in enumerate(zip(k_times_dt, k_indices)):
        axs[0].bar(time, k, width=0.1, color=colors[i], align="center")
    axs[0].set_ylabel("K-index")
    axs[0].set_ylim(0, 9)
    axs[0].grid(True, linestyle="--", alpha=0.7)
    
    if 'dX/dt_smooth' in recent_data.columns:
        axs[1].plot(recent_data["DATETIME"], recent_data["dX/dt_smooth"], label="dX/dt", color="red", linewidth=1)
    if 'dF/dt_smooth' in recent_data.columns:
        axs[1].plot(recent_data["DATETIME"], recent_data["dF/dt_smooth"], label="dF/dt", color="orange", linewidth=1)
    if 'dH/dt_smooth' in recent_data.columns:
        axs[1].plot(recent_data["DATETIME"], recent_data["dH/dt_smooth"], label="dH/dt", color="yellow", linewidth=1)
    axs[1].set_ylabel("Derivatives (nT/min)")
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].legend()
    
    if 'X' in recent_data.columns:
        axs[2].plot(recent_data["DATETIME"], recent_data["X"], label="X", color="red", linewidth=1)
    if 'H' in recent_data.columns:
        axs[2].plot(recent_data["DATETIME"], recent_data["H"], label="H", color="yellow", linewidth=1)
    axs[2].set_ylabel("Magnetic Intensity (nT)")
    axs[2].grid(True, linestyle="--", alpha=0.7)
    axs[2].legend(loc="upper left")
    
    axs[-1].xaxis.set_major_locator(mdates.DayLocator())
    axs[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b'))
    axs[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
    axs[-1].tick_params(axis="x", which="major", pad=15)
    axs[-1].tick_params(axis="x", which="minor", labelsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("ENT_Kindex.png")
    plt.close()

def main_loop():
    plt.ion()
    while True:
        try:
            logging.info("Fetching data...")
            files = get_ftp_files()
            if not files:
                logging.warning("No files downloaded")
                time.sleep(update_interval_minutes * 60)
                continue

            all_data = pd.DataFrame()
            station_name = station_code.upper()
            components = None

            for file_path in files:
                data, comps, name = read_iaga2002(file_path)
                if not data.empty:
                    all_data = pd.concat([all_data, data], ignore_index=True)
                    if components is None:
                        components = comps
                        station_name = name

            if all_data.empty:
                logging.warning("No valid data processed")
                time.sleep(update_interval_minutes * 60)
                continue

            all_data = preprocess_data(all_data, components)
            all_data.sort_values("DATETIME", inplace=True)

            times_float = time_to_float(all_data["DATETIME"])
            comp_x = all_data[components[0]].values
            comp_y = all_data[components[1]].values

            k_indices, k_times = calculate_k_index(times_float, comp_x, comp_y, k9_limit)

            for k, t in zip(k_indices, k_times):
                point = Point("k_index") \
                    .field("value", float(k)) \
                    .time(datetime.utcfromtimestamp(t), WritePrecision.NS)
                write_api.write(bucket=influx_bucket, org=influx_org, record=point)
            logging.info("K-index data written to InfluxDB")

            if len(k_indices) > 0:
                plot_k_indices_with_derivatives(all_data, k_indices, k_times, station_name)

            logging.info(f"Sleeping for {update_interval_minutes} minutes...")
            time.sleep(update_interval_minutes * 60)

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main_loop()
