# --- START OF FILE EthTEC.py ---

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for running as a service
import matplotlib.pyplot as plt
from matplotlib.path import Path
from datetime import datetime, date, timedelta, timezone # <-- MODIFIED: import timezone
import requests
import time
import os

def main():
    print("EthTEC Auto-Refresh System Initialized")
    
    # Create the 'assets' directory if it doesn't exist
    assets_dir = 'assets'
    os.makedirs(assets_dir, exist_ok=True)
    
    while True:
        try:
            # Get current UTC time using the modern, timezone-aware method
            now = datetime.now(timezone.utc) # <-- CORRECTED: Replaced utcnow()
            print(f"\nProcessing TEC data for {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # Time parameters
            year = now.year 
            month = now.month
            day_of_month = now.day
            hour = now.hour + now.minute/60
            longres = .1
            latres = .1
            
            # Calculate day of year
            doy = (date(year, month, day_of_month) - date(year, 1, 1)).days + 1
            
            # Get solar data
            f10p7 = getF10p7N(year, month, day_of_month)
            if f10p7 is None:
                print("Warning: Using default F10.7 value (100.0)")
                f10p7 = 100.0
            
            # --- Data Processing ---
            long_start, long_end = 33, 48
            lat_start, lat_end = 3, 15
            long = np.arange(long_start, long_end + longres, longres)
            lat = np.arange(lat_start, lat_end + latres, latres)
            long2, lat2 = np.meshgrid(long, lat)
            szl, szll = long2.shape, long2.size
            LTHourm = hour + long2.ravel() / 15
            DOYs = np.sin((2 * np.pi * doy) / 365.25)
            DOYc = np.cos((2 * np.pi * doy) / 365.25)
            LTHoursm = np.sin((2 * np.pi * LTHourm) / 24)
            LTHourcm = np.cos((2 * np.pi * LTHourm) / 24)
            inputs = np.column_stack([np.full(szll, DOYc), np.full(szll, DOYs), LTHourcm, LTHoursm, long2.ravel(), lat2.ravel(), np.full(szll, f10p7)])
            TEC = net32D(inputs)
            TEC[TEC < 0] = 0
            TECm = TEC.reshape(szl)
            gmea = pd.read_csv('geomagnetic_equator.txt', sep='\t', skiprows=1, header=None, names=['lon', 'lat'])
            coast = pd.read_csv('Ethiopia_border.txt', sep=',')
            GNSS = pd.read_csv('GNSS_Stn.txt', sep=',')
            points = np.column_stack((long2.ravel(), lat2.ravel()))
            polygon = Path(np.column_stack((coast['Lon'].values, coast['Lat'].values)))
            TECm.ravel()[~polygon.contains_points(points)] = np.nan
            f_L1 = 1575.42e6
            k_L1 = 40.3e16 / f_L1**2
            
            # --- Plotting ---
            plt.close('all')
            vmin = round(np.nanmin(TECm), 0)-10 
            vmax =round(np.nanmax(TECm), 0)+10
            fig = plt.figure(figsize=(11, 7), facecolor='black')
            ax = fig.add_subplot(111, facecolor='white')
            mesh = ax.pcolormesh(long2, lat2, TECm, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            colorbar_axes = divider.append_axes('right', size="8%", pad=0.7)
            cbar = plt.colorbar(mesh, cax=colorbar_axes)
            cbar.set_label(' Vertical TEC (TECU)', fontsize=16, color='white')
            cbar.ax.yaxis.set_label_position('left')
            cbar.ax.yaxis.set_tick_params(color='white')
            tec_ticks = np.linspace(vmin, vmax, 6)
            cbar.set_ticks(tec_ticks)
            cbar.set_ticklabels([f"{tick:.0f}" for tick in tec_ticks], color='white', fontsize=14)
            cbar_ax2 = cbar.ax.twinx()
            l1_ticks = k_L1 * np.linspace(vmin, vmax, 6)
            cbar_ax2.set_ylim(cbar.ax.get_ylim())
            cbar_ax2.set_yticks(tec_ticks)
            cbar_ax2.set_yticklabels([f"{val:.1f}" for val in l1_ticks], color='white', fontsize=14)
            cbar_ax2.set_ylabel("Ionospheric Range Error (m)", fontsize=16, color='white')
            ax.plot(coast['Lon'], coast['Lat'], 'k', linewidth=2)
            ax.plot(gmea['lon'], gmea['lat'], 'k--', linewidth=2)
            if len(GNSS) > 1:
                ax.scatter(GNSS['lon'][1], GNSS['lat'][1], c='r', s=100)
                ax.text(GNSS['lon'][1] - 0.3, GNSS['lat'][1] - 0.5, GNSS['Stn'][1], fontsize=11, weight='bold')
            timestamp = now.strftime('%Y-%m-%d %H:%M UTC')
            
            fname = os.path.join(assets_dir, 'TEC_Map.png')
            
            ax.set_xlim([long_start, long_end])
            ax.set_ylim([lat_start, lat_end])
            ax.set_xlabel('Longitude (Degree)')
            ax.set_ylabel('Latitude (Degree)')
            ax.set_title(f'GPS L1 Range Error From TEC {timestamp}', fontsize=18, color='white')
            ax.axis('off')
            ax.grid(False)
            
            plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='black')
            print(f"Saved image to: {fname}")
            
            plt.close(fig)
            
            next_run = now + timedelta(minutes=1)
            print(f"Next update at: {next_run.strftime('%H:%M UTC')}")
            # Use the corrected 'now' for sleep calculation as well
            sleep_time = max(0, (next_run - datetime.now(timezone.utc)).total_seconds())
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error encountered in EthTEC.py: {str(e)}")
            print("Retrying in 1 minute...")
            time.sleep(60)

# (getF10p7N and net32D functions remain unchanged)
def getF10p7N(year,month,day):
    url='https://services.swpc.noaa.gov/text/27-day-outlook.txt'
    try:
        response=requests.get(url,timeout=10)
        response.raise_for_status()
        lines=response.text.splitlines()
        target_date=datetime(year,month,day).strftime('%Y %b %d')
        for line in lines:
            if line.startswith(target_date):
                return float(line.split()[3])
        return None
    except Exception as e:
        print(f"Error fetching F10.7 data: {e}")
        return None

def net32D(X):
    xoffset_in=np.array([-0.999979192959082,-0.999999422024692,-0.999961923064171,-0.999961923064171,-24.5,-39.5,64.6])
    gain_in=np.array([1.00001502754206,1.00000751371875,1.00003807838574,1.00003807838574,0.0235294117647059,0.025,0.008])
    ymin_in=-1;ymin_out=-1;gain_out=0.0101778702932617;xoffset_out=-1.294764
    b1=np.array([-7.151240997518097,-6.233513633129376,-9.805887729783322,1.4863680183981614,2.8285474831191273,3.355765502836118,2.4332099575126267,-0.6933513899499154,0.3207235332148206,0.06264234274932738,-2.427227218179492,-2.2199511944234525,-0.4543605332993931,-0.019597338846176832,-3.8924470218581773,-0.5599037064658408,0.4877941096074368,0.1005994009386451,-0.8925932028578889,-3.0744175319546665,-5.063686953282598,1.6472738268159786,-0.07341998040622327,-3.4364383585389735,16.118731418067956,0.2742553034310726,2.9156842156572838,-3.1964549688939727,-0.22128518590320395,6.400307356683685,-9.446332937266213,-4.271067250378368])
    b2=-5.44220828420302
    LW1=np.array([[2.3871,-1.758,2.7527,0.2318,-1.2832,1.9932,-1.1744],[-14.745,14.3941,6.9367,6.4649,4.854,1.5765,0.1673],[-6.4331,-8.0666,2.6159,-4.4971,-6.1497,-6.5116,2.42], [0.0423,0.1146,-1.9167,1.2288,0.0134,-0.0479,0.0844],[-0.0838,-0.0565,-0.2417,0.2131,-4.24,2.2364,0.0335],[0.0045,0.0168,-1.4669,-2.3242,0.0194,1.2439,-0.1096],[0.0206,-0.0199,0.2766,-0.2082,-0.3987,-4.8088,0.5051],[-0.2656,0.0475,1.6084,0.36,-0.8291,-4.0363,0.1873],[0.0598,0.0068,0.2659,-0.0667,0.0537,4.2154,0.3308],[-0.2341,-0.6121,-0.0981,0.3717,-0.0958,-0.1815,-0.387],[-4.9545,-1.3622,3.3457,0.3994,-0.6246,-0.5388,-2.5588],[-1.0286,0.4831,-0.2532,0.0095,0.0164,-0.3904,-4.5108],[-0.2031,-0.0966,1.3922,-0.2272,-0.2077,0.1742,-0.3364],[-0.1576,0.5534,0.1277,0.1613,-0.0514,-0.0088,-0.324], [1.3921,0.0662,0.4704,-0.167,0.2996,2.865,-1.8029],[-0.6351,0.2803,0.0377,0.0171,-0.0747,0.0424,0.04],[-0.1483,-0.017,-0.1848,-0.2575,-0.1016,-2.489,0.0378],[-5.9161,-4.3375,4.0143,-2.4277,0.261,-1.4892,6.6889],[0.1035,0.0044,0.0792,0.1605,0.1792,3.1608,-0.0955],[-1.1162,0.1089,1.9674,0.2684,0.188,-3.0699,-1.3453],[-2.1181,4.7815,-2.2497,-1.9518,-0.3287,1.3139,4.0885],[-0.8898,-0.5741,-0.038,-0.1507,0.0412,0.1801,0.2843],[0.0632,-0.173,-0.254,-0.1497,0.0615,-0.0837,0.3814],[3.2873,-1.7928,1.307,7.1192,1.7019,3.5691,-0.6764],[-9.2833,13.4101,1.4171,4.6693,-19.6532,-12.0795,17.9178],[-0.0949,0.0166,-0.0822,0.1215,-0.1481,-4.434,-0.0359],[-0.0026,0.0224,-1.4092,-2.1318,0.0523,0.5723,-0.1194],[-1.1581,0.2339,0.1627,-0.2075,0.19, -5.8542,-1.7279],[-0.1595,0.012,-0.2783,0.0483,0.2141,-1.7495,0.8031],[0.856,0.1855,-1.6038,2.2611,-0.0776,-2.757,0.9777],[-3.8981,0.5471,2.0923,-0.4934,-3.7385,-1.1814,1.2709],[-0.3392,0.0701,-0.1248,-0.2444,0.1798,-0.3903,-4.103]])
    LW2=np.array([0.0637,-0.0098,0.0098,0.4308,-0.057, -2.2412,0.7314,0.1329,1.227,0.482,0.0083,-0.0787,0.5285,2.2643,-0.1951,-1.2594,2.4512,-0.0102,3.1375,-0.2076,0.0403,1.0324,2.7086,-0.0004,0.0018,1.4623,3.4987,-0.1026,-0.4844,-0.3617,-1.6201,-0.3345])
    X=X.T;N=X.shape[1];yy=X-xoffset_in.reshape(-1,1);yy=yy*gain_in.reshape(-1,1);Xp1=yy+ymin_in;n1=np.tile(b1.reshape(-1,1),(1,N))+LW1@Xp1;a1=2/(1+np.exp(-2*n1))-1;n2=np.tile(b2,(1,N))+LW2@a1;a2=2/(1+np.exp(-2*n2))-1;xx2=a2-ymin_out;xx2=xx2/gain_out;Y=xx2+xoffset_out;return Y.T

if __name__ == "__main__":
    main()
