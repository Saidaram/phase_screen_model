# phase_screen_model


Phase Screen Model allows to calculates scintillation indices, for electron density data from the ICI-4 rocket.
This was written for a Master's thesis, and all the codes found here are the main codes used to: filter the measurements from the rocket and calculate the amplitude and phase scintillation indices. 
Also, to allow comparing the indices with ground based receivers, codes such as: skyview_x.py and receivers_data.py were used to: 
1- Locate the best satellite to use, according to the time and location of the ICI-4 rocket. 
2- Study the scintillations receiver's data, from the choosen satellite. And, compare the measurements, from the data calculated by the model. 


\Electron density:

Filtering the electron density data from the ICI-4 rocket is done through ici4_ne_filter.py script.
The code uses the time series as computed by the rocket 'time_data.txt', and a file, containing the cleaned electron density 'ne_clean.txt'. This two files are obtained using the current data measured by the rocket, and, saturated data are interpolated. These files can be obtained by contacting: saida.ramdani@hotmail.com
The file outputs time and filtered densities. The filter used is a notch filter to get rid off the rocket's spin frequency and 8 other harmonics.
There is also a plotting routine, that can be used to display the electron densities before, and after, filtering : plot_ne(t, ne, filtered_ne). This routine requires data from the ICI-4 trajectory that can be obtained by contacting: lasse.clausen@fys.uio.no

\Skyview images:

The skyview images from the different satellites, for different receivers can be performed through skyview_190215_nord_nor.py script. 
Four files are needed for the different receivers at Bjørnøya, Hopen, Longyearbyen and Ny-Ålesund. The files can be obtained by sendind an email to: lasse.clausen@fys.uio.no or w.j.miloch@fys.uio.no
Skibotn receiver station is also used in the study and the code can be obtained by sending an email to: saida.ramdani@hotmail.com
For each receiver:
- the file name need to be changed: L-60: 20150219_filename_REDOBS_gps.txt
- The receiver location need to be changed accordingly: L-119 : r_gps, theta_gps, phi_gps  = receiver2sphere(latitude,longitude,range) 
- The file name for the figure: L-146: plt.title(' ICI-4 rocket path and the different satellites as perceived by {}'.format(receivers[1])) according to the list: receivers = ['BJN', 'HOP', 'NYA', 'LYB']

\Scintillations receivers data: 

The scintillations receivers data are plotted using the script: phase_amplitude_scint_receivers.py
For this script, one can chose the different PRNs (satellites) for the plot, and the phase and scintillation indicex for the different receivers will be shown on the same figure. 
The files for the different receivers can be obtained by sending an email to : lasse.clausen@fys.uio.no or w.j.miloch@fys.uio.no. 
These files are the same files used for the Skyview images. However, in this script ,Skibotn stationmeasurements are included.






