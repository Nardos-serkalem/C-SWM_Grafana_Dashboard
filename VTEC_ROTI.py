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
        for i in range(3)  
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
            'Total_S4_Sig1', 'Correction_total_S4_Sig1', 'Total_S4_Sig2', 'Correction_total_S4_Sig2'
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
        
        return data[['SVID', 'S4_index', 'Dlat_IPP', 'Dlong_IPP', 'VTEC']]
    
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
            # Save processed CSV in the same directory
            csv_filename = os.path.basename(filepath).replace('.ismr', '.csv')
            csv_path = os.path.join(local_dir, csv_filename)
            data.to_csv(csv_path, index=True)
            all_data.append(data)
            #print(f"Processed {filepath} -> {csv_path}")

    if all_data:
        VTEC_ROTI = pd.concat(all_data)
        VTEC_ROTI.sort_index(inplace=True)
        merged_path = os.path.join(local_dir, 'vtec_roti.csv')
        VTEC_ROTI.to_csv(merged_path)
        #print(f"Merged data saved to {merged_path}")
    return VTEC_ROTI


def plot_continuous_timeseries(VTEC_ROTI, bg_color='black', plot_bg='black'):
   
    if not isinstance(VTEC_ROTI.index, pd.DatetimeIndex):
        VTEC_ROTI.index = pd.to_datetime(VTEC_ROTI.index)
    
    # Filter PRNs 1 to 32
    filtered_data = VTEC_ROTI[VTEC_ROTI['SVID'].between(1, 32)]
    
    merged_reset = filtered_data.reset_index().sort_values(['SVID', 'Time'])
    
    # Calculate time differences and filter gaps >60 seconds
    merged_reset['delta_time'] = merged_reset.groupby('SVID')['Time'].diff().dt.total_seconds()
    valid_data = merged_reset[merged_reset['delta_time'] >= 60]
    
    # Calculate proper ROT (TECU/min)
    #valid_data['ROT'] = (valid_data.groupby('SVID')['VTEC'].diff() / valid_data['delta_time']) 
    valid_data.loc[:, 'ROT'] = (valid_data.groupby('SVID')['VTEC'].diff() / valid_data['delta_time'])
    # Calculate ROTI using 5-minute rolling window
    valid_data.set_index('Time', inplace=True)
    valid_data['ROTI'] = (
        valid_data.groupby('SVID')['ROT']
        .rolling('5T', min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    
    # ########figure########################
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))  
    fig.patch.set_facecolor(bg_color)
    for a in ax:
        a.set_facecolor(plot_bg)
    
    # Time range setup
    start_date = filtered_data.index.min().normalize()
    end_date = filtered_data.index.max().normalize() + pd.Timedelta(days=1)
    time_ticks = pd.date_range(start=start_date, end=end_date, freq='6H')
    
    # Collect all handles and labels for the legend (ALL PRNs)
    all_handles = []
    all_labels = []
    
    '''# VTEC plot #####################
    for svid, group in filtered_data.groupby('SVID'):
        sc = ax[0].scatter(group.index, group['VTEC'], label=f'PRN {svid}', s=10)
        all_handles.append(sc)
        all_labels.append(f'PRN {svid}')
    
    mean_vtec = filtered_data.groupby(filtered_data.index)['VTEC'].mean()
    mean_line = ax[0].plot(mean_vtec.index, mean_vtec.values, color='black', linewidth=1.5, label='Mean VTEC')
    all_handles.append(mean_line[0])
    all_labels.append('Mean VTEC')
    
    vtec_max = max(mean_vtec)
    
    # Configure VTEC plot axes
    ax[0].set_xlabel('Time (UT)', fontsize=18)
    ax[0].set_ylabel('VTEC (TECU)', fontsize=18)
    ax[0].set_title('Total Electron Content', fontsize=18, fontweight='bold')
    ax[0].set_xlim(start_date, end_date)
    ax[0].set_ylim(0, vtec_max+20)
    ax[0].set_xticks(time_ticks)
    ax[0].set_xticklabels([t.strftime('%H:%M') for t in time_ticks], rotation=0)
    ax[0].tick_params(axis='x', labelsize=18)
    ax[0].tick_params(axis='y', labelsize=18)
    ax[0].grid(True, linestyle='-.', linewidth=0.5, alpha=0.7)

    # ROTI plot#############################
    for svid, group in valid_data.groupby('SVID'):
        sc = ax[1].scatter(group.index, group['ROTI'], s=5, label=f'{svid}')
        # No need to append again (already in all_handles)
    
    ax[1].set_xlabel('Time (UT)', fontsize=18)
    ax[1].set_ylabel('ROTI (TECU/min)', fontsize=18)
    ax[1].set_title('Rate of TEC Index', fontsize=18, fontweight='bold')
    ax[1].set_ylim(0, 1.0)
    ax[1].set_xlim(start_date, end_date)
    ax[1].set_xticks(time_ticks)
    ax[1].set_xticklabels([t.strftime('%H:%M') for t in time_ticks], rotation=0, fontsize=18)
    ax[1].tick_params(axis='x', labelsize=18)
    ax[1].tick_params(axis='y', labelsize=18)
    ax[1].grid(True, linestyle='--', alpha=0.7)

    # Add day labels
    for mid in pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D') + pd.Timedelta(hours=12):
        ax[1].text(mid, -0.3, mid.strftime('%Y-%m-%d'), horizontalalignment='center', fontsize=18)'''
    for svid, group in filtered_data.groupby('SVID'):
    # Replace gaps with NaN 
       vtec_series = group['VTEC'].reindex(filtered_data.index.unique(), fill_value=np.nan)
       sc = ax[0].scatter(vtec_series.index, vtec_series, label=f'PRN {svid}', s=15)
       all_handles.append(sc)
       all_labels.append(f'PRN {svid}')

    # Calculate mean VTEC with NaN handling
    mean_vtec = filtered_data.groupby(filtered_data.index)['VTEC'].mean()
    mean_vtec = mean_vtec.reindex(filtered_data.index.unique(), fill_value=np.nan)  

    # Plot mean VTEC as separate points (not line) to avoid interpolation
    mean_scatter = ax[0].scatter(mean_vtec.index, mean_vtec, color='blue', s=10, label='Mean VTEC')
    all_handles.append(mean_scatter)
    all_labels.append('Mean VTEC')

    vtec_max = max(mean_vtec.dropna()) if not mean_vtec.dropna().empty else 20  # Fallback if all NaN

    # ROTI plot
    for svid, group in valid_data.groupby('SVID'):
       # Replace gaps with NaN
       roti_series = group['ROTI'].reindex(valid_data.index.unique(), fill_value=np.nan)
       ax[1].scatter(roti_series.index, roti_series, s=15, label=f'{svid}')

    # Configure both plots
    for i, (title, ylabel, ymax) in enumerate(zip(
        #[f'ENTG GNSS Total Electron Content\n Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}', 'Rate of TEC Index']
       [f"ENTG GNSS Total Electron Content: Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 'Rate of TEC Index'],
       ['VTEC (TECU)', 'ROTI (TECU/min)'], [vtec_max+20, 1.0])):
      ax[i].set_xlabel('Time (UT)', fontsize=18, color='white')
      ax[i].set_ylabel(ylabel, fontsize=16, color='white')
      ax[i].set_title(title, fontsize=18, fontweight='bold', color='white')
      ax[i].set_xlim(start_date, end_date)
      ax[i].set_ylim(0, ymax)
      ax[i].set_xticks(time_ticks)
      ax[i].set_xticklabels([t.strftime('%H:%M') for t in time_ticks], rotation=0, color='white')
      ax[i].tick_params(axis='both', labelsize=18,colors='white')
      ax[i].grid(True, linestyle='--', alpha=0.7)
     

    # Add day labels
    '''for mid in pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D') + pd.Timedelta(hours=12):
        ax[1].text(mid, -0.3, mid.strftime('%Y-%m-%d'), ha='center', fontsize=16, color='white')
    '''# Create a unified vertical legend outside the plot (ALL PRNs)
    '''fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc='center right',
        fontsize=11,
        ncol=1,  
        bbox_to_anchor=(0.98, 0.5),  # Adjust position to avoid overlap
        frameon=True,
        title='GPS PRNs',
        title_fontsize=16,
    )'''

    # Adjust layout to make room for the legend

    # Adjust layout to make room for the legend
    plt.subplots_adjust(right=0.95)  # Reduce right margin to fit legend
    
    plt.tight_layout()
    plt.savefig(os.path.join(local_dir, 'ENTG_VTEC_and_ROTI.png'), edgecolor='black', facecolor=fig.get_facecolor())
    plt.close()
def main():
    while True:
        print("Checking for new ISMR files...")
        download_ismr_files()
        remove_old_ismr_files()  
        
        ismr_files = glob.glob(os.path.join(local_dir, "*.ismr"))
        
        if ismr_files:
            VTEC_ROTI = process_ismr_files()
            plot_continuous_timeseries(VTEC_ROTI)
        else:
            print("No new ISMR files found")

        print("Waiting 3 minute before next check...")
        time.sleep(180) 

if __name__ == "__main__":
    main()
