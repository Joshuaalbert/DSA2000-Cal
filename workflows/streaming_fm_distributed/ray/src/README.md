The scaling properties off the streaming forward model are a simple multiplicative chain.

We can perform the scaling analysis per sub-band of 1000 channels, i.e. 25 spectral solution intervals. 
Horizontal scaling thereafter is possible.

Aggregator (1 per subband, Low FLOP, sums gridded images and PSFs and saves)
-> Gridder
num_cpus=1
mem=
gpu_mem=0
run_time=
num_actors=1
total_resource_req=1 CPUs

Gridder (1 per solint, High FLOP, wgridding of subtracted visibilities)
-> Calibrator
num_cpus=32 (160 total threads per call)
mem=
gpu_mem=0
run_time=
num_actors=25 (num_channels / num_chans_per_sol_int)
total_resource_req=32 * 25 CPUs = 800 CPUs, 

Calibrator (1 per solint, High FLOP, calibrates data against model on CPU or GPU t.b.d.)
-> DataStreamer
-> ModelPredictor
num_cpus=1 t.b.d.
mem=
gpu_mem=0
run_time=
num_actors=25 (num_channels / num_chans_per_sol_int)
total_resource_req=1 * 25 CPUs = 25 CPUs, 

DataStreamer (1 per time/freq, Low FLOP, orchestrates gain model and predictor actors)
-> SystematicGainSimulator
-> DFTPredictor
-> DegriddingPredictor
num_cpus=0
mem=
gpu_mem=0
run_time=
num_actors=25*4*40=4000 (sol_ints*times_per_sol_int*freqs_per_sol_int)
total_resource_req=0 CPUs, 

ModelPredictor (1 per time/freq, Low FLOP, orchestrates gain model and predictor actors)
-> DFTPredictor
-> DegriddingPredictor
num_cpus=0
mem=
gpu_mem=0
run_time=
num_actors=25*4*40=4000 (sol_ints*times_per_sol_int*freqs_per_sol_int)
total_resource_req=0 CPUs, 

SystematicGainSimulator (1 per time/freq, Medium FLOP, computes gain model at certain time and frequency quickly using interpolation)
num_cpus=1
mem=
gpu_mem=0
run_time=
num_actors=4000 (1/DataStreamer)
total_resource_req=0 CPUs, 

DFTPredictor (2 per time/freq/coh, does DFT predict of points and gaussian components on GPU for fast evaluation)
num_cpus=0
mem=
gpu_mem=0.1
run_time=
num_actors=10000 (num_channels)
total_resource_req=1000 GPUs, 

DegriddingPredictor (2 per time/freq/coh, does wgridder degridding on CPU)
num_cpus=coh (4)
mem=
gpu_mem=0
run_time=
num_actors=10000 (num_channels)
total_resource_req=0 CPUs, 

## Aggregator



# Gridder