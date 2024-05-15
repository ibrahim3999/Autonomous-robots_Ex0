# Autonomous-robots_Ex0

This repository contains code for processing GNSS (Global Navigation Satellite System) measurements, specifically for GPS (Global Positioning System) satellites. The code is written in Python and involves processing raw GNSS measurements to determine satellite positions and correct measured pseudorange values.

## Overview

GNSS measurements are obtained from a receiver device, usually in the form of raw logs. These logs contain information such as pseudorange, carrier frequency, signal strength (C/N0), and Doppler shift for each satellite visible to the receiver.

The main steps involved in the processing are:

1. **Reading and Preprocessing**: Reading raw GNSS logs from a file, preprocessing the data, and converting timestamps to GPS and Unix time formats.

2. **Satellite Position Calculation**: Using ephemeris data (orbital information of satellites) and measured pseudoranges, satellite positions are calculated in Earth-Centered Earth-Fixed (ECEF) coordinates.

3. **Least Squares Estimation**: A least squares estimation method is employed to refine satellite positions and correct clock biases.

4. **Geodetic Conversion**: Converting satellite positions from ECEF to geodetic coordinates (latitude, longitude, altitude).

5. **Visualization and Output**: Outputting processed data in various formats, such as CSV files and KML files for visualization in Google Earth.

## Usage

1. Clone this repository to your local machine:

   ```bash
   git https://github.com/ibrahim3999/Autonomous-robots_Ex0.git

2.    install packages
      ```bash
   $ pip install -r requirements.txt

3.
   ```bash 
   python gnss_to_csv.py data/file_gnss_name.txt


## Requirements

1. python 3.x
2. pandas
3. numpy
4. matplotlib
5. lxml
6. simplekml
7. navpy
8. gnssutils


## Visualization Screenshot

This screenshot shows the driving path visualized in Google Earth:

![Driving Path in Google Earth](https://github.com/ibrahim3999/Autonomous-robots_Ex0/blob/main/GNSS_Raw_Mesurments/data/driving.png)
