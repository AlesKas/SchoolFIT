all:
	nvcc template.cu -O3 -o ex2 -I../libwb/ ../libwb/lib/libwb.a
#	nvcc solution.cu -O3 -o ex1 -I../libwb/ ../libwb/lib/libwb.a
	g++ dataset_generator.cpp -O3 -fopenmp -o generator -I../libwb/ ../libwb/lib/libwb.a

gen:
	./generator

run0:
	./ex2 -e TextHistogram/Dataset/0/output.raw -i TextHistogram/Dataset/0/input.txt -o TextHistogram/Dataset/0/out.raw -t integral_vector
                
run1:
	./ex2 -e TextHistogram/Dataset/1/output.raw -i TextHistogram/Dataset/1/input.txt  -t integral_vector

run2:
	./ex2 -e TextHistogram/Dataset/2/output.raw -i TextHistogram/Dataset/2/input.txt  -t integral_vector

run3:
	./ex2 -e TextHistogram/Dataset/3/output.raw -i TextHistogram/Dataset/3/input.txt  -t integral_vector

run4:
	./ex2 -e TextHistogram/Dataset/4/output.raw -i TextHistogram/Dataset/4/input.txt  -t integral_vector

run5:
	./ex2 -e TextHistogram/Dataset/5/output.raw -i TextHistogram/Dataset/5/input.txt  -t integral_vector


PROFILE=/apps/all/CUDAcore/11.1.1/nsight-compute-2020.2.1/target/linux-desktop-glibc_2_11_3-x64/ncu --export ./Example2_profile \
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
	$(PROFILE) ./ex2 -e TextHistogram/Dataset/0/output.raw -i TextHistogram/Dataset/0/input.txt -t integral_vector

prof1:	
	$(PROFILE) ./ex2 -e TextHistogram/Dataset/1/output.raw -i TextHistogram/Dataset/1/input.txt  -t integral_vector

prof2:	
	$(PROFILE) ./ex2 -e TextHistogram/Dataset/2/output.raw -i TextHistogram/Dataset/2/input.txt  -t integral_vector

prof3:	
	$(PROFILE) ./ex2 -e TextHistogram/Dataset/3/output.raw -i TextHistogram/Dataset/3/input.txt  -t integral_vector

prof4:	
	$(PROFILE) ./ex2 -e TextHistogram/Dataset/4/output.raw -i TextHistogram/Dataset/4/input.txt  -t integral_vector

prof5:	
	$(PROFILE) ./ex2 -e TextHistogram/Dataset/5/output.raw -i TextHistogram/Dataset/5/input.txt  -t integral_vector




clean:
	rm -f ex2 generator
