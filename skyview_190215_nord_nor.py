#This program displays the skyview images for different PRN's at different receivers stations
import pandas as pd
import numpy as np 
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import math


prn = [31.0, 9.0, 10.0, 2.0, 29.0,  20.0, 23.0, 25.0]

def sphere2coord(r, phi, theta):

	latitude = 90 - theta
	longitude = phi

	return latitude, longitude

def xyz2sphere(x1,y1,z1,x2,y2,z2):

	x, y, z = x1+x2, y1+y2, z1+z2

	r = np.degrees(np.sqrt(x**2 + y**2 + z**2))	
	theta = np.degrees(np.arctan((np.sqrt(x**2 + y**2))/z))	
	phi = np.degrees(np.arctan2(y,x))

	return r, theta, phi

def sphere2xyz(r, phi, theta):

	x = r * np.cos(np.radians(phi)) * np.sin(np.radians(theta))
	y = r * np.sin(np.radians(phi)) * np.sin(np.radians(theta))
	z = r * np.cos(np.radians(theta))

	return x, y, z

def aer2sphere(azimuth, elevation, Range, phi_gps):

	R = Range
	theta = 90 - elevation
	phi = 200 - azimuth

	return R, theta, phi

def receiver2sphere(latitude, longitude, elevation):

	earth_radius = 6378.1370 # km
	R = earth_radius + elevation
	theta = 90 - latitude
	phi = longitude

	return R, theta, phi

def string_to_float(data, name):
	return np.array(((data[name]).to_numpy()),dtype=np.float32)

def gps(prn):

	data = pd.read_csv('uib_scint/20150219_HOP_REDOBS_gps.txt',\
	skiprows=17,header=0, error_bad_lines=False,sep =",") # Change HOP to the right receiver

	L1_band = data.loc[data[' SigType'] == 1]

	t1 = np.where(L1_band[' HHMM'] == 2130)[0][0]
	t2 = np.where(L1_band[' HHMM'] == 2329)[0][-1]
	
	dt = L1_band[t1:t2]
	dt = dt.loc[dt[' PRN'] == prn] # Choose prn 

	time = string_to_float(dt, ' HHMM')
	az = string_to_float(dt, ' Az ')
	el = string_to_float(dt, ' Elv ') 
 
	return time, az, el

df = pd.read_csv('ici4_actual_trajectory_10Hz.dat',  skiprows = 2, header=0, sep ="\s+",names=['Time', 'Alt', 'Lat', 'Lon', 'Vel', 'Vel. El', 'Vel. Az'],low_memory=False)
geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
gdf = GeoDataFrame(df, geometry=geometry)

#Map of Norway
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world[(world.name == "Norway")].plot(figsize=(10, 6), color='white', edgecolor='black'), marker='_', color='black', markersize=7, label='ICI-4')#[(world.continent == "Europe")]
plt.plot(19.001, 74.504,  marker='+', color='red', markersize=6)
plt.plot(25.014, 76.509,  marker='+', color='red', markersize=6)
plt.plot(16.038, 78.147,  marker='+', color='red', markersize=6)
plt.plot(11.929, 78.923,  marker='+', color='red', markersize=6)
plt.plot(20.38, 69.43,  marker='+', color='red', markersize=6)


colors = ['chocolate','fuchsia','lime','red', 'grey','yellow','blue','cyan','indigo']
width = [2, 6, 3, 2,2, 1, 2, 2 ]

lon_list = []
lat_list = []

for number in range(len(prn)):
	
	t, azimuth, elevation = gps(prn[number])

	index = np.where(t == 2206)[0][0]
	
	markers_on = np.arange(index,index+11,1)

	if prn[number] == 9.0:
		avg = [np.round((360+((azimuth[-1]-360)+azimuth[1]) / 2),0),np.round(np.average(elevation))]

	elif prn[number] == 20.0:
		avg = [np.round((((azimuth[-1]-360)+azimuth[1]) / 2),0),np.round(np.average(elevation))]


	else:
		avg = [np.round(np.average(azimuth)),np.round(np.average(elevation))]

	# Latitude, Longitude, range for different receivers
	# BJN: (74.504, 19.001, 0.026)  HOP: (76.509,25.014,0.014) 
	# KHO: (78.147,16.038,0.522)  NYA: (78.923,11.929,0.021)

	r_gps, theta_gps, phi_gps  = receiver2sphere(76.509,25.014,0.014) # Change according to the receiver's location

	r_rock, theta_rock, phi_rock  = aer2sphere(azimuth,elevation, 350, phi_gps)
	
	x1, y1, z1 = sphere2xyz(r_gps, phi_gps, theta_gps)
	x2, y2, z2 = sphere2xyz(r_rock, phi_rock, theta_rock)

	r_f, theta_f, phi_f = xyz2sphere(x1,y1,z1,x2,y2,z2)
	final_lat, final_lon = sphere2coord(r_f, phi_f, theta_f)

	lat_list.append(final_lat)	
	lon_list.append(final_lon)

	fig = plt.gcf()
	plt.plot(final_lon, final_lat, linewidth=width[number],  marker='D', color=colors[number], markersize=3, markevery=markers_on, markeredgecolor='black',label = 'PRN {}, az/el {}'.format(prn[number], avg))
	plt.plot(final_lon[-1], final_lat[-1], 'kp',markersize=3)

receivers = ['BJN', 'HOP', 'NYA', 'LYB']

plt.annotate('BJN', (19.001,74.504))
plt.annotate('HOP', (25.014, 76.509))
plt.annotate('LYB', (16.038, 78.147))
plt.annotate('NYA', (11.929, 78.923))
plt.annotate('SKI', (20.38, 69.43))
plt.xlabel('Longitude [Degrees]')
plt.ylabel('Latitude [Degrees]')

plt.title(' ICI-4 rocket path and the different satellites as perceived by {}'.format(receivers[1]))

l = plt.legend(ncol=1, fontsize=5, frameon = 1, facecolor='white', framealpha=1)
l.set_draggable(True)
plt.grid()
plt.show()