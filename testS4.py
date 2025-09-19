import os
import pandas as pd
import numpy as np
import ftplib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from scipy.signal import medfilt
import logging
import time
from ftplib import FTP, all_errors
import tempfile
from scipy.stats import zscore
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib
matplotlib.use('Agg')

from prometheus_client import Gauge, start_http_server

plt.rcParams.update({'font.size': 12})

# ================= Logging ================= #
logging.basicConfig(
    filename='ismr_download.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================= Prometheus Metrics ================= #
s4_index_metric = Gauge('ismr_s4_index', 'S4 Scintillation Index', ['svid'])
vtec_metric = Gauge('ismr_vtec', 'Vertical TEC', ['svid'])

# ================= FTP Settings ================= #
FTP_HOST = "gnssdatacenter.strela.rssi.ru"
FTP_DIR = "/gnss_data/ismr/ENTG/"
LOCAL_DIR = "E:/ismr"
STATION_CODE = "ENTG"
DATE = datetime(2024, 9, 5)  # Example date

# ================= Helper Functions ================= #
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def download_files(ftp, remote_dir, local_dir, station_code, date):
    ftp.cwd(remote_dir)
    ensure_directory_exists(local_dir)

    remote_filename = f"{station_code}{date.strftime('%y%j')}.ismr"
    local_filename = os.path.join(local_dir, remote_filename)

    if os.path.exists(local_filename):
        logging.info(f"File already exists: {local_filename}")
        return local_filename

    with open(local_filename, "wb") as f:
        ftp.retrbinary(f"RETR {remote_filename}", f.write)
    logging.info(f"Downloaded {remote_filename} to {local_filename}")
    return local_filename

def parse_ismr_file(filepath):
    data = []
    with open(filepath, "r") as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                date_str, time_str = parts[0], parts[1]
                datetime_str = date_str + " " + time_str
                timestamp = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")
                svid = int(parts[2])
                azimuth = float(parts[3])
                elevation = float(parts[4])
                cn0 = float(parts[5])
                s4_index = float(parts[6])
                sigma_phi = float(parts[7])
                vtec = float(parts[8])

                data.append([timestamp, svid, azimuth, elevation, cn0, s4_index, sigma_phi, vtec])
            except Exception as e:
                logging.warning(f"Error parsing line: {line} - {e}")

    return pd.DataFrame(data, columns=["Timestamp", "SVID", "Azimuth", "Elevation", "CN0", "S4_index", "Sigma_phi", "VTEC"])

def process_ismr_data(filepath, output_dir):
    df = parse_ismr_file(filepath)
    if df.empty:
        logging.warning("Parsed DataFrame is empty")
        return None, None

    output_csv = os.path.join(output_dir, f"{STATION_CODE}_processed.csv")
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved processed data to {output_csv}")

    return df, output_csv

def plot_s4(df, output_dir):
    plt.figure(figsize=(12, 6))
    for svid in df["SVID"].unique():
        svid_data = df[df["SVID"] == svid]
        plt.plot(svid_data["Timestamp"], svid_data["S4_index"], label=f"SVID {svid}")

    plt.xlabel("Time")
    plt.ylabel("S4 Index")
    plt.title("S4 Index over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_png = os.path.join(output_dir, f"{STATION_CODE}_S4_index.png")
    plt.savefig(output_png)
    plt.close()
    logging.info(f"Saved S4 index plot to {output_png}")
    return output_png

def update_prometheus_metrics(df):
    for _, row in df.iterrows():
        svid = str(row["SVID"])
        s4_index_metric.labels(svid=svid).set(row["S4_index"])
        vtec_metric.labels(svid=svid).set(row["VTEC"])

# ================= Main ================= #
def main():
    try:
        ftp = FTP(FTP_HOST, timeout=120)
        ftp.login()
        logging.info("Connected to FTP server")

        local_file = download_files(ftp, FTP_DIR, LOCAL_DIR, STATION_CODE, DATE)
        ftp.quit()

        if local_file:
            df, csv_file = process_ismr_data(local_file, LOCAL_DIR)
            if df is not None:
                plot_s4(df, LOCAL_DIR)
                update_prometheus_metrics(df)
                logging.info("Updated Prometheus metrics")

    except all_errors as e:
        logging.error(f"FTP error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    start_http_server(8000)  # Expose metrics on http://localhost:8000/metrics
    while True:
        main()
        time.sleep(300)  # run every 5 minutes
