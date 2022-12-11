How to use samples:

- use sampleInput.h5 file as an input to your simulation.

- run simulation with these parameters 
	N=4096
	DT=0.01f
	STEPS=2500
	WRITE_INTESITY=50 

- compare the output of the simulation against sampleOutput.h5 using command 
	h5diff -v2 -p [R] sampleOutput.h5 [simulation_output] /[dataset_name] 
	where R is a relative error (should be approx. in the order of 10^-7) 
	simulation_output is the output of your simulation and dataset_name is 
	the dataset you wish to compare (note leading forward slash)

- steps 0 - 2
	compare only datasets pos_x_final, pos_y_final, pos_z_final, weight_final,
	vel_x_final, vel_y_final and vel_z_final. No other dataset is implemented 
	yet.

- step 3
	you may start compare datasets com_x_final, com_y_final, com_z_final 
	and com_w_final

- step 4 
	compare all datasets
