all:
	nvcc template.cu -O3 -o ex1 -I./inc 
#	nvcc solution.cu -O3 -o ex1 -I./inc 


run1:
	./ex1 -input data/lena.pgm -reference data/lena_ref.pgm

run2:
	./ex1 -input data/landscape.pgm -reference data/landscape_ref.pgm

PROFILE=/apps/all/CUDAcore/11.1.1/nsight-compute-2020.2.1/target/linux-desktop-glibc_2_11_3-x64/ncu --export ./Example1_profile \
			--force-overwrite \
			--target-processes application-only \
			--replay-mode kernel \
			--kernel-regex-base function \
			--launch-skip-before-match 0 \
			--section ComputeWorkloadAnalysis \
			--section InstructionStats \
			--section LaunchStats \
			--section MemoryWorkloadAnalysis \
			--section MemoryWorkloadAnalysis_Chart \
			--section MemoryWorkloadAnalysis_Tables \
			--section Occupancy \
			--section SchedulerStats \
			--section SourceCounters \
			--section SpeedOfLight \
			--section SpeedOfLight_RooflineChart \
			--section WarpStateStats \
			--sampling-interval auto \
			--sampling-max-passes 5 \
			--sampling-buffer-size 33554432 \
			--profile-from-start 1 \
			--cache-control all \
			--clock-control base \
			--apply-rules yes \
			--check-exit-code yes


prof1:
	$(PROFILE) ./ex1 -input data/lena.pgm -reference data/lena_ref.pgm 

prof2:
	$(PROFILE) ./ex1 -input data/landscape.pgm -reference data/landscape_ref.pgm

clean:
	rm -f ex1
