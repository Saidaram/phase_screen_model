# phase_screen_model


Phase Screen Model allows to calculates scintillation indices, for electron density data from the ICI-4 rocket.
This was written for a Master's thesis, and all the codes found here are the main codes used to: Test the developped method, filter the measurements from the rocket, and, calculate the amplitude and phase scintillation indices. 
Also, to allow comparing the indices with ground based receivers, codes such as: skyview_x.py and receivers_data.py were used to: 
1- Locate the best satellite to use, according to the time and location of the ICI-4 rocket. 
2- Study the scintillations receiver's data, from the choosen satellite. And, compare the measurements, from the data calculated by the model. 


\Electron density:

Filtering the electron density data from the ICI-4 rocket is done throug ici4_ne_filter.py script.
The code uses the time series as computed by the rocket 'time_data.txt', and a file, containing the cleaned electron density 'ne_clean.txt'. This two files are obtained using the current data measured by the rocket, and, saturated data are interpolated. 
The file outputs time and filtered densities. The filter used is a notch filter to get rid off the rocket's spin frequency and 8 other harmonics.

There is also a plotting routine, that can be used to display the electron densities before, and after, filtering : plot_ne(t, ne, filtered_ne)




