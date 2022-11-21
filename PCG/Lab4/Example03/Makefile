all:
	nvcc template.cu -O3 -o ex3 -I../libwb/ ../libwb/lib/libwb.a
#	nvcc solution.cu -O3 -o ex1 -I../libwb/ ../libwb/lib/libwb.a
	g++ dataset_generator.cpp -O3 -fopenmp -o generator -I../libwb/ ../libwb/lib/libwb.a

gen:
	./generator

run0:
	./ex3 -e Reduction/Dataset/0/output.raw -i Reduction/Dataset/0/input.raw  -t vector

run1:
	./ex3 -e Reduction/Dataset/1/output.raw -i Reduction/Dataset/1/input.raw  -t vector

run2:
	./ex3 -e Reduction/Dataset/2/output.raw -i Reduction/Dataset/2/input.raw  -t vector

run3:
	./ex3 -e Reduction/Dataset/3/output.raw -i Reduction/Dataset/3/input.raw  -t vector

run4:
	./ex3 -e Reduction/Dataset/4/output.raw -i Reduction/Dataset/4/input.raw  -t vector

run5:
	./ex3 -e Reduction/Dataset/5/output.raw -i Reduction/Dataset/5/input.raw  -t vector


PROFILE=/apps/all/CUDAcore/11.1.1/nsight-compute-2020.2.1/target/linux-desktop-glibc_2_11_3-x64/ncu --export ./Example3_profile \
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



prof0:	
	$(PROFILE) ./ex3 -e Reduction/Dataset/0/output.raw -i Reduction/Dataset/0/input.raw -t vector

prof1:	
	$(PROFILE) ./ex3 -e Reduction/Dataset/1/output.raw -i Reduction/Dataset/1/input.raw  -t vector

prof2:	
	$(PROFILE) ./ex3 -e Reduction/Dataset/2/output.raw -i Reduction/Dataset/2/input.raw  -t vector

prof3:	
	$(PROFILE) ./ex3 -e Reduction/Dataset/3/output.raw -i Reduction/Dataset/3/input.raw  -t vector

prof4:	
	$(PROFILE) ./ex3 -e Reduction/Dataset/4/output.raw -i Reduction/Dataset/4/input.raw -t vector

prof5:	
	$(PROFILE) ./ex3 -e Reduction/Dataset/5/output.raw -i Reduction/Dataset/5/input.raw -t vector




clean:
	rm -f ex3 generator
