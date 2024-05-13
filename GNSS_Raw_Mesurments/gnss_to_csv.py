import sys
import traceback
import os
import csv
import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
from pykml.factory import KML_ElementMaker as KML
import navpy
import simplekml
from gnssutils import EphemerisManager

parent_directory = os.getcwd()
ephemeris_data_directory = os.path.join(parent_directory, 'data')

pd.options.mode.chained_assignment = None

# Constants
WEEKSEC = 604800
lightSpeed = 2.99792458e8
gpsepoch = datetime(1980, 1, 6, 0, 0, 0)


def least_squares(xs, measured_pseudorange, x0, b0):
    dx = 100 * np.ones(3)
    b = b0
    # set up the G matrix with the right dimensions. We will later replace the first 3 columns
    # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0
    while np.linalg.norm(dx) > 1e-3:
        # Eq. (2):
        r = np.linalg.norm(xs - x0, axis=1)
        # Eq. (1):
        phat = r + b0
        # Eq. (3):
        deltaP = measured_pseudorange - phat
        G[:, 0:3] = -(xs - x0) / r[:, None]
        # Eq. (4):
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
        # Eq. (5):
        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db
    norm_dp = np.linalg.norm(deltaP)
    return x0, b0, norm_dp


def read_data(input_filepath):
    measurements, android_fixes = [], []
    with open(input_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    return pd.DataFrame(measurements[1:], columns=measurements[0])


def preprocess_measurements(measurements):
    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
    # measurements['Constellation'] = measurements['ConstellationType'].map({'1': 'G', '3': 'R'})
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'

    # Create SvName and filter GPS satellites only
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']
    measurements = measurements[measurements['Constellation'] == 'G']

    # Convert columns to numeric representation and handle missing data robustly
    numeric_cols = ['Cn0DbHz', 'TimeNanos', 'FullBiasNanos', 'ReceivedSvTimeNanos',
                    'PseudorangeRateMetersPerSecond', 'ReceivedSvTimeUncertaintyNanos',
                    'BiasNanos', 'TimeOffsetNanos']
    for col in numeric_cols:
        measurements[col] = pd.to_numeric(measurements[col], errors='coerce').fillna(0)

    # Generate GPS and Unix timestamps
    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (
                measurements['FullBiasNanos'] - measurements['BiasNanos'])
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)

    # Identify epochs based on time gaps
    measurements['Epoch'] = 0
    time_diff = measurements['UnixTime'] - measurements['UnixTime'].shift()
    measurements.loc[time_diff > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()

    # Calculations related to GNSS Nanos, week number, seconds, pseudorange
    measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - \
                                   (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9 * measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9 * (measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

    # Convert pseudorange from seconds to meters
    measurements['PrM'] = lightSpeed * measurements['prSeconds']
    measurements['PrSigmaM'] = lightSpeed * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

    return measurements


def ecef_to_geodetic(x, y, z):
    # WGS84 ellipsoid constants
    a = 6378137.0  # Earth's radius in meters
    e_sq = 6.69437999014e-3  # Eccentricity squared

    # Calculations
    b = np.sqrt(a ** 2 * (1 - e_sq))  # Semi-minor axis
    ep = np.sqrt((a ** 2 - b ** 2) / b ** 2)
    p = np.sqrt(x ** 2 + y ** 2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep ** 2 * b * np.sin(th) ** 3, p - e_sq * a * np.cos(th) ** 3)
    N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert radian to degrees for latitude and longitude
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def calculate_satellite_position(ephemeris, transmit_time):
    mu = 3.986005e14
    OmegaDot_e = 7.2921151467e-5
    F = -4.442807633e-10
    sv_position = pd.DataFrame()
    sv_position['sv'] = ephemeris.index
    sv_position.set_index('sv', inplace=True)
    sv_position['t_k'] = transmit_time - ephemeris['t_oe']
    A = ephemeris['sqrtA'].pow(2)
    n_0 = np.sqrt(mu / A.pow(3))
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['t_k']
    E_k = M_k
    err = pd.Series(data=[1] * len(sv_position.index))
    i = 0
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e'] * np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1

    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)
    delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
    delT_oc = transmit_time - ephemeris['t_oc']
    sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris[
        'SVclockDriftRate'] * delT_oc.pow(2)

    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e']))

    Phi_k = v_k + ephemeris['omega']

    sin2Phi_k = np.sin(2 * Phi_k)
    cos2Phi_k = np.cos(2 * Phi_k)

    du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
    dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
    di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

    u_k = Phi_k + du_k

    r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k

    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['t_k']

    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)

    Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * ephemeris[
        't_oe']

    sv_position['x_k'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
    sv_position['y_k'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
    sv_position['z_k'] = y_k_prime * np.sin(i_k)
    return sv_position


def save_kml(geodetic_positions):
    kml = simplekml.Kml()
    for index, row in geodetic_positions.iterrows():
        kml.newpoint(name=f"Satellite {index}", coords=[(row['Longitude'], row['Latitude'], row['Altitude'])])
    kml.save("satellite_positions.kml")


def write_outputs(ecef_positions, kml_filepath):
    # Create KML for visualization
    doc = KML.kml(
        KML.Document(
            *[KML.Placemark(
                KML.name(str(i)),
                KML.Point(KML.coordinates(f"{pos[1]},{pos[0]},{pos[2]}"))
            ) for i, pos in enumerate(ecef_positions)]
        )
    )

    # Write KML file
    print("write kml")
    with open(kml_filepath, 'wb') as file:
        file.write(etree.tostring(doc, pretty_print=True))


def main():
    # Options to choose from the datasets
    parser = argparse.ArgumentParser(description="Process GNSS log file to compute paths and save to KML.")
    parser.add_argument("file_path", type=str, help="Path to the GNSS log file")
    args = parser.parse_args()
    parsed_measurements = read_data(args.file_path)
    measurements = preprocess_measurements(parsed_measurements)
    manager = EphemerisManager(ephemeris_data_directory)

    csvoutput = []
    ecef_list = []
    for epoch in measurements['Epoch'].unique():
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
        if len(one_epoch.index) > 4:
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)

            # Calculating satellite positions (ECEF)
            sats = one_epoch.index.unique().tolist()
            ephemeris = manager.get_ephemeris(timestamp, sats)
            sv_position = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

            # Apply satellite clock bias to correct the measured pseudorange values
            # Ensure sv_position's index matches one_epoch's index
            sv_position.index = sv_position.index.map(str)  # Ensuring index types match; adjust as needed
            one_epoch = one_epoch.join(sv_position[['delT_sv']], how='left')
            pr = one_epoch['PrM_Fix'] = one_epoch['PrM'] + lightSpeed * one_epoch['delT_sv']
            pr = pr.to_numpy()

            # Doppler shift calculation
            doppler_calculated = False
            try:
                one_epoch['CarrierFrequencyHz'] = pd.to_numeric(one_epoch['CarrierFrequencyHz'])
                one_epoch['DopplerShiftHz'] = -(one_epoch['PseudorangeRateMetersPerSecond'] / lightSpeed) * one_epoch[
                    'CarrierFrequencyHz']
                doppler_calculated = True
            except Exception:
                pass

            b0 = 0
            x0 = np.array([0, 0, 0])
            xs = sv_position[['x_k', 'y_k', 'z_k']].to_numpy()
            x, b, dp = least_squares(xs, pr, x0, b0)
            ecef_list.append(x[:3])
            for sv in one_epoch.index:
                csvoutput.append({
                    "GPS Time": timestamp.isoformat(),
                    "SatPRN (ID)": sv,
                    "Sat.X": sv_position.at[sv, 'x_k'],
                    "Sat.Y": sv_position.at[sv, 'y_k'],
                    "Sat.Z": sv_position.at[sv, 'z_k'],
                    "Pseudo-Range": one_epoch.at[sv, 'PrM_Fix'],
                    "CN0": one_epoch.at[sv, 'Cn0DbHz'],
                    "Doppler": one_epoch.at[sv, 'DopplerShiftHz'] if 'DopplerShiftHz' in one_epoch.columns else np.nan,
                    "Pos.X": x[0],
                    "Pos.Y": x[1],
                    "Pos.Z": x[2],
                })

    print(navpy.ecef2lla(x))
    print(b / lightSpeed)
    print(dp)
    ecef_array = np.stack(ecef_list, axis=0)
    # print(ecef_array)
    lla_array = np.stack(navpy.ecef2lla(ecef_array), axis=1)
    # print("checking")
    print(lla_array)
    write_outputs(lla_array, 'path.kml')
    ref_lla = lla_array[0, :]
    ned_array = navpy.ecef2ned(ecef_array, ref_lla[0], ref_lla[1], ref_lla[2])
    # print(ned_array)
    coordinates = [ecef_to_geodetic(row['x_k'], row['y_k'], row['z_k']) for index, row in sv_position.iterrows()]
    geodetic_positions = pd.DataFrame(coordinates, columns=['Latitude', 'Longitude', 'Altitude'])
    # print(geodetic_positions)
    csv_df = pd.DataFrame(csvoutput)
    save_kml(geodetic_positions)
    # Append geodetic positions to CSV output
    csv_df['Latitude'] = geodetic_positions['Latitude']
    csv_df['Longitude'] = geodetic_positions['Longitude']
    csv_df['Altitude'] = geodetic_positions['Altitude']

    csv_df.to_csv("gnss_measurements_output.csv", index=False)


try:
    main()
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()  # This prints the stack trace to the standard error
