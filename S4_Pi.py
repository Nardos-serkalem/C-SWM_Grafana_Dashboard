######Daniel Chekole#########
import os
import ftplib
import gzip
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import glob
import subprocess
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend like 'Agg'
#matplotlib.use('TkAgg')  # or 'Qt5Agg'
#os.environ["QT_QPA_PLATFORM"] = "xcb"  # Use X11 instead of Wayland

from prometheus_client import start_http_server, Gauge

# Define Prometheus metrics
s4_index_gauge = Gauge('s4_index', 'S4 scintillation index', ['svid'])
vtec_gauge = Gauge('vtec', 'Vertical Total Electron Content', ['svid'])
phi60_gauge = Gauge('phi60', 'Sigma Phi (60s detrended)', ['svid'])

# Start Prometheus server on port 8000
start_http_server(8000)
print("Prometheus exporter running on http://localhost:8000/metrics")


# FTP server credentials
ftp_server = "ftp.gnss.sansa.org.za"
username = "ngiday"
password = "j8dheeZJ"
# Local directory for ISMR files
local_dir = r"/home/space/Downloads/spaceWheatherFinal/spaceWheather/ismr/assets/"
local_dir = os.path.join(os.getcwd(), "assets")
os.makedirs(local_dir, exist_ok=True)

# Function to get the last three days
def get_last_three_days():
    today = datetime.utcnow()
    return [
        {"year": (today - timedelta(days=i)).strftime("%Y"),
         "month": (today - timedelta(days=i)).strftime("%m"),
         "day": (today - timedelta(days=i)).strftime("%d"),
         "yy": (today - timedelta(days=i)).strftime("%y"),
         "doy": (today - timedelta(days=i)).timetuple().tm_yday}
        for i in range(8)  
    ]

# Function to remove old ISMR files (older than the last three days)
def remove_old_ismr_files():
    days_info = get_last_three_days()
    last_three_doys = {day["doy"] for day in days_info}

    for filename in os.listdir(local_dir):
        if filename.startswith("ENTG"):
            try:
                doy = int(filename[4:7])  # Extract DOY from filename
                if doy not in last_three_doys:
                    file_path = os.path.join(local_dir, filename)
                    os.remove(file_path)
                    #print(f"Removed old file: {file_path}")
            except ValueError:
                continue
                
                if doy not in last_three_doys:
                    file_path = os.path.join(folder, filename)
                    os.remove(file_path)
                    #print(f"Removed old ISMR file: {file_path}")
   
# Function to download ISMR files
def download_ismr_files():
    days_info = get_last_three_days()[:3]  
    try:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login(user=username, passwd=password)
            print("Connected to FTP server.")

            for day_info in days_info:
                folder = f"/home/ethiopiagnss/ENTGST1/R/ismr/{day_info['year']}/{day_info['month']}/{day_info['day']}/"
                print(f"Checking folder: {folder}")

                try:
                    ftp.cwd(folder)
                    items = ftp.nlst()

                    for item in items:
                        if item.endswith(".ismr.gz"):
                            local_file = item.replace(".gz", "")
                            local_path = os.path.join(local_dir, local_file)

                            if os.path.exists(local_path):
                                #print(f"File already exists: {local_file}. Skip downloading.")
                                continue

                            gz_path = local_path + ".gz"
                            with open(gz_path, "wb") as f:
                                ftp.retrbinary(f"RETR {item}", f.write)
                            print(f"Downloaded: {item}")

                            extract_gz(gz_path, local_path)

                except ftplib.error_perm as e:
                    print(f"Failed to access folder {folder}: {e}")

    except Exception as e:
        print(f"FTP connection failed: {e}")

# Function to extract .gz files
def extract_gz(gz_path, target_path):
    try:
        with gzip.open(gz_path, 'rb') as f_in, open(target_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"Extracted: {target_path}")
    except gzip.BadGzipFile:
        print(f"Warning: Bad GZIP file detected for {gz_path}. Skip extraction.")


##########################read ISMR###############
def read_ismr(filename, lat='9.11', lon='38.79', columns=None, Ipp=350, skiprows=None, dtype=None, elevation_mask=20):
    
    ismr_column = [
        'GPS_Week_Number', 'GPS_Time_Week', 'SVID', 'Value', 'Azimuth', 'Elevation',
        'Sig1', 'Total_S4_Sig1', 'Correction_total_S4_Sig1', 'Phi01_Sig1_1',
        'Phi03_Sig1_3', 'Phi10_Sig1_10', 'Phi30_Sig1_30', 'Phi60_Sig1_60',
        'AvgCCD_Sig1_average_code-carrier_divergence',
        'SigmaCCD_Sig1_standard_deviation_code-carrier_divergence',
        'TEC_TOW-45s', 'dTEC_TOW-60s_TOW-45s', 'TEC_TOW-30s',
        'dTEC_TOW-45s_TOW-30s', 'TEC_TOW-15s', 'dTEC_TOW-30s_TOW-15s',
        'TEC_TOW', 'dTEC_TOW-15s_TOW', 'Sig1_lock_time',
        'sbf2ismr_version_number', 'Lock_time_second_frequency_TEC',
        'Averaged_C/N0_second_frequency_TEC_computation', 'SI_Index_Sig1',
        'SI_Index_Sig1_numerator', 'p_Sig1_spectral_slope',
        'Average_Sig2_C/N0', 'Total_S4_Sig2', 'Correction_total_S4_Sig2',
        'Phi01_Sig2_1', 'Phi03_Sig2_3', 'Phi10_Sig2_10', 'Phi30_Sig2_30',
        'Phi60_Sig2_60', 'AvgCCD_Sig2_average_code-carrier_divergence',
        'SigmaCCD_Sig2_standard', 'Sig2_lock', 'SI_Index_Sig2',
        'SI_Index_Sig2_numerator', 'p_Sig2_phase',
        'Average_Sig3_C/N0_last_minute', 'Total_S4_Sig3',
        'Correction_total_S4_Sig3', 'Phi01_Sig3_1_phase', 'Phi03_Sig3_3_phase',
        'Phi10_Sig3_10_phase', 'Phi30_Sig3_30_phase', 'Phi60_Sig3_60_phase',
        'AvgCCD_Sig3_average_code-carrier_divergence',
        'SigmaCCD_Sig3_standard_deviation_code-carrier_divergence',
        'Sig3_lock_time', 'SI_Index_Sig3', 'SI_Index_Sig3_numerator',
        'p_Sig3_phase', 'T_Sig1_phase', 'T_Sig2_phase', 'T_S3_phase'
    ]
    
    if columns is not None:
        ismr_column = columns

    try:
        data = pd.read_csv(
            filename,
            names=ismr_column,
            skiprows=skiprows,
            dtype=str  
        )
        
        # Convert numerical columns to proper types
        numeric_cols = [
            'GPS_Week_Number', 'GPS_Time_Week', 'SVID', 'Azimuth', 'Elevation', 
            'Total_S4_Sig3', 'Correction_total_S4_Sig3', 'TEC_TOW', 
            'Total_S4_Sig1', 'Correction_total_S4_Sig1', 'Total_S4_Sig2', 'Correction_total_S4_Sig2', 'Phi60_Sig1_60'
        ]
        
        # Ensure all columns that should be numeric are converted
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # elevation mask (>20 degrees)
        data = data[data['Elevation'] >= elevation_mask]
        
        # Convert GPS time to UTC datetime
        data['Time'] = data.apply(lambda row: __weeksecondstoutc(row['GPS_Week_Number'], row['GPS_Time_Week']), axis=1)
        data = data.set_index('Time')

        # Compute S4 index
        data['S4_index_1'] = np.sqrt(data['Total_S4_Sig1']**2 - data['Correction_total_S4_Sig1']**2)
        data['S4_index'] = np.round(data['S4_index_1'] * 100) / 100
        data['S4_index'][data['S4_index'] > 3] = np.nan

        # Compute VTEC from STEC using provided formula
        Re = 6371  # Mean Earth radius in km
        hs = 350   # Thin-shell effective altitude in km
        data['Sf'] = (1 - ((Re * np.cos(np.radians(data['Elevation']))) / (Re + hs))**2)**(-0.5)
        data['VTEC'] = data['TEC_TOW'] / data['Sf']


        # Compute Ionospheric Pierce Point (IPP)
        PHI = float(lat)
        LAMBDA = float(lon)
        ELEV = np.deg2rad(data['Elevation'])
        AZI = np.deg2rad(data['Azimuth'])
        RE = 6378136.3  # Earth radius in meters
        IPP = Ipp * 1000  # Convert to meters
        Iono_ht = (RE / (RE + IPP)) * np.cos(ELEV)
        Shi_pp = (np.pi / 2) - ELEV - np.arcsin(Iono_ht)
        Phi_pp = np.arcsin(np.sin(np.deg2rad(PHI)) * np.cos(Shi_pp) + np.cos(np.deg2rad(PHI)) * np.sin(Shi_pp) * np.cos(AZI))
        Lambda_pp = np.deg2rad(LAMBDA) + np.arcsin(np.sin(Shi_pp) * np.sin(AZI) / np.cos(Phi_pp))
        
        data['Dlat_IPP'] = np.rad2deg(Phi_pp)
        data['Dlong_IPP'] = np.rad2deg(Lambda_pp)
        data['Stec'] = data['TEC_TOW']
        
        return data[['SVID', 'S4_index', 'Dlat_IPP', 'Dlong_IPP', 'VTEC', 'Phi60_Sig1_60']]
    
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None
def __weeksecondstoutc(gpsweek, gpsseconds):
    """Convert GPS week and seconds to UTC time."""
    import datetime
    gpsweek = float(gpsweek)
    gpsseconds = float(gpsseconds)
    epoch = datetime.datetime(1980, 1, 6)
    elapsed = datetime.timedelta(days=(gpsweek * 7), seconds=gpsseconds)
    return epoch + elapsed


def process_ismr_files():
    all_data = []
    input_pattern = os.path.join(local_dir, "*.ismr")

    for filepath in glob.glob(input_pattern):
        data = read_ismr(filepath)
        if data is not None:
            csv_filename = os.path.basename(filepath).replace('.ismr', '.csv')
            csv_path = os.path.join(local_dir, csv_filename)
            data.to_csv(csv_path, index=True)
            all_data.append(data)

            # Export metrics for each satellite (SVID)
            for _, row in data.iterrows():
                svid = str(row['SVID'])
                if not np.isnan(row['S4_index']):
                    s4_index_gauge.labels(svid=svid).set(row['S4_index'])
                if not np.isnan(row['VTEC']):
                    vtec_gauge.labels(svid=svid).set(row['VTEC'])
                if not np.isnan(row['Phi60_Sig1_60']):
                    phi60_gauge.labels(svid=svid).set(row['Phi60_Sig1_60'])

    if all_data:
        S4_pi = pd.concat(all_data)
        S4_pi.sort_index(inplace=True)
        merged_path = os.path.join(local_dir, 'S4_pi_roti.csv')
        S4_pi.to_csv(merged_path)
        return S4_pi
    return pd.DataFrame()

def plot_continuous_timeseries(S4_pi, bg_color='black', plot_bg='black'):
    # Ensure datetime index
    if not isinstance(S4_pi.index, pd.DatetimeIndex):
        S4_pi.index = pd.to_datetime(S4_pi.index)


    # Filter GPS PRNs 1-32 and valid data
    filtered_data = S4_pi[(S4_pi['SVID'].between(1, 32)) & 
                    (S4_pi['S4_index'].notna()) & 
                    (S4_pi['Phi60_Sig1_60'].notna())].copy()
    
    # Create time-ordered data
    filtered_data.sort_index(inplace=True)
    
    # Time range setup
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))
    fig.patch.set_facecolor(bg_color)
    for a in ax:
        a.set_facecolor(plot_bg)
    start_date = filtered_data.index.min().normalize()
    end_date = filtered_data.index.max().normalize() + pd.Timedelta(days=1)
    time_ticks = pd.date_range(start=start_date, end=end_date, freq='6H')
    # ----------------- Plot 1: S4 Index -------------------
    '''for prn in filtered_data['SVID'].unique():
        prn_data = filtered_data[filtered_data['SVID'] == prn]
        ax[0].scatter(prn_data.index, prn_data['S4_index'], s=5, label=f'PRN {prn}')
    
    ax[0].set_title('GPS S4 Scintillation Index (PRN 1-32)', fontsize=14)
    ax[0].set_ylabel('S4 Index', fontsize=12)
    ax[0].set_ylim(0, 1.2)
    ax[0].legend(loc='upper right', ncol=4, fontsize=8)
    ax[0].set_xticks(time_ticks)'''
    
    # S4 index scatter plot
    colors = filtered_data['S4_index'].apply(lambda x: 'b' if x <= 0.5 else ('g' if x <= 0.8 else 'r'))
    ax[0].scatter(filtered_data.index, filtered_data['S4_index'], c=colors, s=5)
    ax[0].set_xlabel('Time (UT)', fontsize=18, color='white')
    ax[0].set_ylabel('S4', fontsize=18, color='white')
    ax[0].set_title('Amplitude Scintillation ', fontsize=18, fontweight='bold', color='white')
    ax[0].set_xlim(start_date, end_date)
    ax[0].set_ylim(0, 1.5)
    ax[0].set_xticks(time_ticks)
    ax[0].set_xticklabels([t.strftime('%H:%M') for t in time_ticks], rotation=0, color='white')
    ax[0].tick_params(axis='x', labelsize=18, color='white')
    ax[0].tick_params(axis='y', labelsize=18, color='white')
    ax[0].tick_params(axis='both', labelsize='18', colors='white')
    ax[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # S4 legend
    ax[0].scatter([], [], color='b', label='Low (S4 < 0.5)', s=40)
    ax[0].scatter([], [], color='g', label='Moderate (0.5 < S4 < 0.8)', s=40)
    ax[0].scatter([], [], color='r', label='Strong (S4 > 0.8)', s=40)
    ax[0].legend(loc='upper right', fontsize=14)

    
    # ----------------- Plot 2: Sigma_Phi (60s detrended) -------------------
    colors = filtered_data['Phi60_Sig1_60'].apply(lambda x: 'b' if x <= 0.4 else ('g' if x <= 0.7 else 'r'))
    ax[1].scatter(filtered_data.index, filtered_data['Phi60_Sig1_60'], c=colors, s=5)
    ax[1].set_xlabel('Time (UT)', fontsize=14)
    ax[1].set_ylabel(r'$\sigma_{\phi} (1 min)$', fontsize=18, color='white')
    ax[1].set_title('Phase Scintillation', fontsize=18, fontweight='bold', color='white')
    ax[1].set_xlim(start_date, end_date)
    ax[1].set_ylim(0, 1.5)
    ax[1].set_xticks(time_ticks)
    ax[1].set_xticklabels([t.strftime('%H:%M') for t in time_ticks], rotation=0,  color='white')
    ax[1].tick_params(axis='x', labelsize=18, color='white')
    ax[1].tick_params(axis='y', labelsize=18, color='white')
    ax[1].tick_params(axis='both', labelsize='18', colors='white')
    ax[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # sigpi legend
    ax[1].scatter([], [], color='b', label=r'Low ($\sigma_{\phi} < 0.4$)', s=40)
    ax[1].scatter([], [], color='g', label=r'Moderate ($0.4 < \sigma_{\phi} < 0.7$)', s=40)
    ax[1].scatter([], [], color='r', label=r'Strong ($\sigma_{\phi} > 0.7$)', s=40)

    ax[1].legend(loc='upper right', fontsize=14)
    # Add day labels
    for mid in pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D') + pd.Timedelta(hours=12):
        ax[1].text(mid, -0.3, mid.strftime('%Y-%m-%d'), horizontalalignment='center', fontsize=18, color='white')

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(local_dir, 'ENTG_S4_pi.png'), edgecolor='white', facecolor=fig.get_facecolor())
    plt.close()


def main():
    while True:
        print("Checking for new ISMR files...")
        download_ismr_files()
        remove_old_ismr_files()  
        
        ismr_files = glob.glob(os.path.join(local_dir, "*.ismr"))
        
        if ismr_files:
            S4_pi = process_ismr_files()
            if not S4_pi.empty:
                plot_continuous_timeseries(S4_pi)
            else:
                print("No valid data to plot")
        else:
            print("No new ISMR files found")

        print("Waiting 3 minutes before next check...")
        time.sleep(180)

if __name__ == "__main__":
    main()