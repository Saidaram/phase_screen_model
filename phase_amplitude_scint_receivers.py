import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import ConnectionPatch


prn = [31.0, 9.0, 10.0, 2.0, 29.0, 20.0, 23.0, 25.0]
def string_to_float(data, name):
	return np.array(((data[name]).to_numpy()),dtype=np.float32)


def gps_bjn(prn):
	data = pd.read_csv('uib_scint/20150219_BJN_REDOBS_gps.txt',\
	skiprows=17,header=0, error_bad_lines=False,sep =",")
	print(list(data.columns))

	L1_band = data.loc[data[' SigType'] == 1]

	t1 = np.where(L1_band[' HHMM'] == 2151)[0][0]
	t2 = np.where(L1_band[' HHMM'] == 2251)[0][-1]

	dt = L1_band[t1:t2]

	dt = dt.loc[dt[' PRN'] == prn] # Choose prn 
	#print(dt)

	time = string_to_float(dt, ' HHMM')
	s4 = string_to_float(dt, ' S4 ') - string_to_float(dt, ' S4 Cor')
	sigma_60s = string_to_float(dt, ' 60SecSigma ') 

	return time, s4, sigma_60s


t, s4, sig60 = gps_bjn(prn[0])

idx = np.where(t == 2209)[0][0]
h = np.arange(0,len(t),1)

t1_psm = (np.ones_like(h))*h[idx]
t2_psm = (np.ones_like(h))*h[idx+4]


fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
# Amplitude scintillations
ax1.plot(h, s4, 'o-', color = 'blue', label ='Bjørnøya')
ax1.set_title(r'$S_4$ index and $\sigma_\phi$ index from GPS L1 for PRN {}'.format(prn[0]))
ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
ax1.set_ylabel(r'$S_4$')

# Set the zoom
sub_s4 = ax1.inset_axes([.15,.5,.2,.35]) #xmin, ymin, dx, dy
s4_region = s4[idx:idx+5]
h_region = h[idx:idx+5]
sub_s4.plot(h_region, s4_region, '*-', color = 'blue')
sub_s4.set_xticklabels('')

# Phase scintillations
ax2.plot(h, sig60, 'o-', color = 'blue', label ='Bjørnøya')
ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
ax2.set_ylabel(r'$\sigma_\phi$ radians ')
ax2.set_xlabel('Time [HHMM]')

# Set the zoom
sub_sig = ax2.inset_axes([.2,.5,.2,.35]) #xmin, ymin, dx, dy
sig_region = sig60[idx:idx+5]
sub_sig.plot(h_region, sig_region, '*-', color = 'blue')
sub_sig.set_xticklabels('')



def gps_hop(prn):

	data = pd.read_csv('uib_scint/20150219_HOP_REDOBS_gps.txt',\
	skiprows=17,header=0, error_bad_lines=False,sep =",")

	L1_band = data.loc[data[' SigType'] == 1]

	t1 = np.where(L1_band[' HHMM'] == 2151)[0][0]
	t2 = np.where(L1_band[' HHMM'] == 2251)[0][-1]

	dt = L1_band[t1:t2]

	dt = dt.loc[dt[' PRN'] == prn] # Choose prn 


	time = string_to_float(dt, ' HHMM')
	s4 = string_to_float(dt, ' S4 ') - string_to_float(dt, ' S4 Cor')
	sigma_60s = string_to_float(dt, ' 60SecSigma ') 


	return time, s4, sigma_60s

t, s4, sig60 = gps_hop(prn[0])

idx = np.where(t == 2209)[0][0]
h = np.arange(0,len(t),1)

t1_psm = (np.ones_like(h))*h[idx]
t2_psm = (np.ones_like(h))*h[idx+4]

ax1.plot(h, s4, 'o-', color = 'green', label ='Hopen')
ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
ax1.set_ylabel(r'$S_4$')

s4_region = s4[idx:idx+5]
sub_s4.plot(h_region, s4_region, '*-', color = 'green')
sub_s4.set_xticklabels('')

ax2.plot(h, sig60, 'o-', color = 'green', label ='Hopen')
ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
ax2.set_ylabel(r'$\sigma_\phi$ radians ')
ax2.set_xlabel('Time [HHMM]')

sig_region = sig60[idx:idx+5]
sub_sig.plot(h_region, sig_region, '*-', color = 'green')

def gps_kho(prn):

	data = pd.read_csv('uib_scint/20150219_KHO_REDOBS_gps.txt',\
	skiprows=17,header=0, error_bad_lines=False,sep =",")

	L1_band = data.loc[data[' SigType'] == 1]

	t1 = np.where(L1_band[' HHMM'] == 2151)[0][0]
	t2 = np.where(L1_band[' HHMM'] == 2251)[0][-1]

	dt = L1_band[t1:t2]

	dt = dt.loc[dt[' PRN'] == prn] # Choose prn 

	time = string_to_float(dt, ' HHMM')
	s4 = string_to_float(dt, ' S4 ') - string_to_float(dt, ' S4 Cor')
	sigma_60s = string_to_float(dt, ' 60SecSigma ') 


	return time, s4, sigma_60s

t, s4, sig60 = gps_kho(prn[0])

idx = np.where(t == 2209)[0][0]
h = np.arange(0,len(t),1)

t1_psm = (np.ones_like(h))*h[idx]
t2_psm = (np.ones_like(h))*h[idx+4]

ax1.plot(h, s4, 'o-', color = 'red', label ='Longyearbyen')
ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
plt.ylabel(r'$S_4$')

s4_region = s4[idx:idx+5]
sub_s4.plot(h_region, s4_region, '*-', color = 'red')
sub_s4.set_xticklabels('')

ax2.plot(h, sig60, 'o-', color = 'red', label ='Longyearbyen')
ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
plt.ylabel(r'$\sigma_\phi$ radians ')
plt.xlabel('Time [HHMM]')

sig_region = sig60[idx:idx+5]
sub_sig.plot(h_region, sig_region, '*-', color = 'red')
sub_sig.set_xticklabels('')

def gps_nya(prn):

	data = pd.read_csv('uib_scint/20150219_NYA_REDOBS_gps.txt',\
	skiprows=17,header=0, error_bad_lines=False,sep =",")

	L1_band = data.loc[data[' SigType'] == 1]

	t1 = np.where(L1_band[' HHMM'] == 2151)[0][0]
	t2 = np.where(L1_band[' HHMM'] == 2251)[0][-1]

	dt = L1_band[t1:t2]

	dt = dt.loc[dt[' PRN'] == prn] # Choose prn 

	time = string_to_float(dt, ' HHMM')
	s4 = string_to_float(dt, ' S4 ') - string_to_float(dt, ' S4 Cor')
	sigma_60s = string_to_float(dt, ' 60SecSigma ') 


	return time, s4, igma_60s

t, s4, sig60 = gps_nya(prn[0])


idx = np.where(t == 2209)[0][0]
h = np.arange(0,len(t),1)

t1_psm = (np.ones_like(h))*h[idx]
t2_psm = (np.ones_like(h))*h[idx+4]


ax1.plot(h, s4, 'o-', color = 'black', label ='Ny-Ålesund')
ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
ax1.set_ylabel(r'$S_4$')

s4_region = s4[idx:idx+5]
sub_s4.plot(h_region, s4_region, '*-', color = 'black')
sub_s4.set_xticklabels('')
ax1.indicate_inset_zoom(sub_s4, edgecolor="black")

ax2.plot(h, sig60, 'o-', color = 'black', label ='Ny-Ålesund')

ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
plt.ylabel(r'$\sigma_\phi$ radians ')
plt.xlabel('Time [HHMM]')
l = plt.legend(fontsize="x-small")
l.set_draggable(True)

sig_region = sig60[idx:idx+5]
sub_sig.plot(h_region, sig_region, '*-', color = 'black')
sub_sig.set_xticklabels('')
ax2.indicate_inset_zoom(sub_sig, edgecolor="black")



# Skibotn
f = open('skibotn/GPS4_ISM_skn_2015-02.asc', 'r') 
lines = f.readlines()[371:]

x = []
for line in lines:
	'''
	First get only the values from 19/02/2015
	'''
	s = line.split(' ')
	day_index = s[0].split('T')

	if (day_index[0] == '2015-02-19')
		last = s[-1].split('\n')
		new_s = s[:-1] + last
		x.append(new_s)
	else:
		pass		
else:
	pass

new_x = []
for a_list in x:
	'''
	Seperate time and date
	'''

	time_index = a_list[0].split('T')
	a_list.pop(0)
	s = list(np.concatenate((time_index,a_list), axis=0))
	new_s = []
	for item in s:
		if item != '':
			new_s.append(item)

	new_x.append(new_s)

for count, liste in enumerate(new_x):
	if liste[1] == '21:51:45.000Z':
		index1 = count
	elif liste[1] == '22:52:45.000Z':
		index2= count

ski_1hour = new_x[10899:11414]
az = []
el = []
s4 = []
sig_phi = []
prn_ski = []

for liste in ski_1hour:
	az.append(float(liste[2]))
	el.append(float(liste[3]))
	s4.append(float(liste[5])-float(liste[6]))
	sig_phi.append(float(liste[11]))
	prn_ski.append(float(liste[-1]))

az = np.array(az)
el = np.array(el)
s4 = np.array(s4)
sig_phi = np.array(sig_phi)
prn_ski = np.array(prn_ski)

prn_uniq = list(set(prn_ski))

prn_mask = (prn_ski == prn[0])

s4 = s4[prn_mask]

sig60 = sig_phi[prn_mask]

idx = np.where(t == 2209)[0][0]
h = np.arange(0,len(t),1)

t1_psm = (np.ones_like(h))*h[idx]
t2_psm = (np.ones_like(h))*h[idx+4]


ax1.plot(h, s4, 'o-', color = 'yellow', label ='Skibotn')
ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
ax1.set_ylabel(r'$S_4$')

s4_region = s4[idx:idx+5]
sub_s4.plot(h_region, s4_region, '*-', color = 'yellow')
sub_s4.set_xticklabels('')
ax1.grid(axis='y')

ax2.plot(h, sig60, 'o-', color = 'yellow', label ='Skibotn')

ax = plt.gca()
ax.set_xticks(h[::5])
ax.set_xticklabels(t[::5])
plt.ylabel(r'$\sigma_\phi$ radians ')
plt.xlabel('Time [HHMM]')
l = plt.legend(fontsize="x-small")
l.set_draggable(True)

sig_region = sig60[idx:idx+5]
sub_sig.plot(h_region, sig_region, '*-', color = 'yellow')
sub_sig.set_xticklabels('')
ax2.indicate_inset_zoom(sub_sig, edgecolor="black")
ax2.grid(axis='y')

plt.show()
