import numpy as np 
from scipy.constants import find, physical_constants, c
import ici4_ne_filter
import matplotlib.pyplot as plt
import datetime

class var:

	re = physical_constants['classical electron radius'][0] # Electron radius [m]
	freq = 1575420000 # [hz] L1 frequency for GNSS cycle/s
	wavelength = c / freq  # [m]
	z = 350000 # Altitude 
	fresnel_length = np.sqrt(wavelength*z) # [m]
	wavenumber = 2 * np.pi / wavelength # [1/m] 
	width_ps = 20000 # Phase screen width [m] #Changeable parameter

def data():
	time_all, electron_density =  ici4_ne_filter.main() # [s] , [electron/m**3]


	# Choosing the right timing where Rocket 
	# is in the F layer 241.34 km and 281.43 
	t0 = 150 # Start time
	tf = 450.443 # End time

	index_t0 = np.where(time_all == t0)[0][0] # Initial time index
	index_tf = np.where(time_all == tf)[0][5] # End time index

	time = time_all[:index_tf]

	return index_t0, index_tf, time, electron_density

def receivers_res():

	# Receiver's resolution

	# Find steps between data
	fs_receiver = 50 # receivers fs
	osd = 1/fs_receiver # time for one-single-data receiver(osd)

	v_iono = 500 # m/s (Verified by superDARN)
	d_osd = v_iono * osd # [m] Distance at which each data is measured

	fs_rocket = 8680 # Hz
	osd_rock = 1/fs_rocket # time for one-single-data rocket(osd)

	v_rocket = 607 # m/s (Measured by rocket trajectory data)
	dosd_rock= v_rocket * osd_rock #[m] dist at which data is measured

	# Number of rocket data for 1 data from receivers on ground
	steps = int(d_osd * 1 / dosd_rock) # Calculated before

	return steps

def phase_screen(d, E_0, ne, x0): 

	#width_ps = 500 # Phase screen width [m] #Changeable parameter
	phi = var.wavelength * var.re * ne * var.width_ps # Phase change
	amplitude_on_ground = 0 

	for x in range(len(d)): # At the phase screen
		amplitude_on_ground += -1*((1j*E_0[x])/(var.wavelength * np.sqrt(var.z)))*\
									np.exp(1j*( var.wavenumber*var.z + phi[x] + \
										((var.wavenumber*((x0-d[x])**2))/(2*var.z))))

	amplitude = (np.array(np.abs(amplitude_on_ground)))
	phase = np.angle(amplitude_on_ground) # Rad

	return amplitude, phase

def intensity_phase(t0, time, steps, ne):

	amp_list = [] # Empty list of amplitudes
	int_list = [] # Empty list of Intensities
	phase_list = [] # Empty list of phases
	time_list = [] # Empty time list

	nbr_data = 16001 # Number of data for 770 m phase screen size (+1 symmetry)

	for i in range(t0, len(time), steps):
		last_idx = i+nbr_data

		t = time[i:last_idx] 

		if len(t) != nbr_data:
			continue

		ne_part = ne[i:last_idx]
		half = int(nbr_data/2)
	
		# Calculate Distances: 
		distance = t*607 # [m]
		x0 = distance[half]
	
		E_0 = np.ones_like(distance) # Initial amplitude of the signal
	
		E, phase = phase_screen(distance, E_0, ne_part, x0)

		time_list.append(t[half])
		amp_list.append(E)
		E_int = E**2 # Intensity as described by Fremouw
		int_list.append(E_int)
		phase_list.append(phase)

	return int_list, phase_list, time_list


def scint_index(time_list,int_list,phase_list ):

	t_steps = np.mean(np.diff(time_list))

	time_bins = np.array([1,3,10,40,60])
	nbOfData = np.int_(time_bins/t_steps) 
	n = nbOfData[-1]+1 # Change the time bins here

	t_list = []
	s4_list = [] # Empty list of s4
	sigma_list = []  # Empty list of sigma phi

	for i,_ in enumerate(time_list[::n]):
		sub_tlist = time_list[i*n:] if (i+1)*n > len(time_list) else time_list[i*n:(i+1)*n] 
		t_list.append(np.mean(sub_tlist))

		sub_alist = int_list[i*n:] if (i+1)*n > len(int_list) else int_list[i*n:(i+1)*n] 
		sub_alist = (np.array(sub_alist))**2
		s4 = np.sqrt( np.var(sub_alist) / (np.mean(sub_alist))**2 )
		s4_list.append(s4)

		sub_plist = phase_list[i*n:] if (i+1)*n > len(phase_list) else phase_list[i*n:(i+1)*n] 
		sigma_phi = np.sqrt(np.var(sub_plist))
		sigma_list.append(sigma_phi)

	return t_list, s4_list, sigma_list

def main():


	t0_idx, tf_idx, time, ne = data()
	steps = receivers_res()
	intensity_list, phase_list, time_list = intensity_phase(t0_idx, time, steps, ne)
	t, s4, sig_phi = scint_index(time_list, intensity_list, phase_list)
	
	return t, s4, sig_phi


if __name__ == "__main__":
	main()
