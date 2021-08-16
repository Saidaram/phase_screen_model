# This code return Notch fiiltered electron density data
# Measured by the ICI-4 rocket
# And a plot of the Data before and after filtering
import numpy as np 
import pandas as pd
from statistics import mean
from scipy import constants as cst
from scipy import signal
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import freqz, lfilter, welch
from scipy import fftpack
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist


def fft_function(sig, fs):

	dt_freq = np.fft.rfftfreq(n = len(sig), d =1/fs)
	dt_amp = np.fft.rfft(sig)

	return dt_freq, dt_amp

def string_to_float(data, name):
    return np.array(((data[name]).to_numpy()), dtype=np.float32)


def plot_ne(t, ne, filtered_ne):
	t0 = 150 # Start time  241342.5 altitude km
	tf = 450.443 # End time 450.50  281427.8 

	index_t0 = np.where(t == t0)[0][0] # Initial time index
	index_tf = np.where(t == tf)[0][5] # End time index

	#Trajectory data
	data_traj = pd.read_csv('ici4_actual_trajectory_10Hz.dat', skiprows = 2, header=0, sep ="\s+",names=['Time', 'Alt', 'Lat', 'Lon', 'Vel', 'Vel. El', 'Vel. Az'])
	time = string_to_float(data_traj, "Time")
	altitude = string_to_float(data_traj, "Alt")

	tt = np.arange(150, 500, 50)

	alt_ticks = []
	for it in tt: 

		index = np.where(time == it)[0][-1]
		alt_ticks.append(altitude[index])

	alt_ticks = np.array(alt_ticks)
	alt_ticks = np.round((alt_ticks/1000),1)

	# Plot the data
	ax  = plt.subplot(211)
	ax.plot(t[index_t0:index_tf], ne[index_t0:index_tf] * 10**(-11), 'b')
	ax.set_ylabel('Electron density\n'+r'[$10^{11} m^{-3}$]')
	ax.set_title('Electron density measured by the m-NLP on the ICI-4 rocket')
	plt.grid()
	ax1 = plt.subplot(212)
	ax1.plot(t[index_t0:index_tf], np.abs(filtered_ne[index_t0:index_tf]) * 10 **(-11), 'b')
	ax1.set_xlim(150,450)
	ax1.set_ylabel('Electron density\n'+r'[$10^{11} m^{-3}$]')
	ax1.set_xlabel(u'Time [s]')
	plt.grid()

	ax2 = ax1.twiny()
	ax2.set_xticks(range(7))
	ax2.set_xticklabels(alt_ticks)

	ax2.xaxis.set_ticks_position('bottom') 
	ax2.xaxis.set_label_position('bottom') 
	ax2.spines['bottom'].set_position(('outward', 36))
	ax2.set_xlabel("Altitude [km]")

	plt.show()

	return None

def main():

	tdata = pd.read_csv('time_data.txt', header=0, sep ="\s+")

	interp_ne = pd.read_csv('ne_clean.txt', header=0, sep ="\s+")
	data_array, time_array = interp_ne.to_numpy(), tdata.to_numpy()
	tt = np.reshape(time_array, len(time_array)) #time data
	ne = np.reshape(data_array, len(data_array))
	t = tt[1:-1]
	
	# Filtering data:
	
	diff_list = np.diff(t) # Find diff between each element in a list:
	T = np.mean(diff_list) #Find the period 
	samp_freq = 1/T # Frequency 1/T (Sampling rate)

	f, w = fft_function(ne, samp_freq)
	power = np.abs(w) ** 2 
	
	#Notch frequency
	fn = 3.196667280432113
	
	list_fn = fn*np.arange(1,10,1)
	list_ind = []
	for freq_n in list_fn:
		ind = np.where(np.isclose(f,freq_n))[0][0]
		list_ind.append(ind)

	# for a broader filter
	idx = np.array(list_ind)
	new_list_ind = []
	for val in idx:
		new_idx = np.arange(val-9, val+10, 1)
		new_list_ind.extend(new_idx)

	w[new_list_ind] = 0

	filtered_ne = np.real_if_close(np.fft.irfft(w))

	#plot_ne(t, ne, filtered_ne)

	return t, filtered_ne

if __name__ == "__main__":
	main()