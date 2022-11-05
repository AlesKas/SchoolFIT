all:
	nvcc template.cu -O3 -o ex1 -I../libwb/ ../libwb/lib/libwb.a
#	nvcc solution.cu -O3 -o ex1 -I../libwb/ ../libwb/lib/libwb.a
	g++ dataset_generator.cpp -O3 -fopenmp -o generator -I../libwb/ ../libwb/lib/libwb.a

gen:
	OMP_NUM_THREADS=2 ./generator

run0:
	./ex1 -e MatrixMultiplication/Dataset/0/output.raw -i MatrixMultiplication/Dataset/0/input0.raw,MatrixMultiplication/Dataset/0/input1.raw -t matrix

run1:
	./ex1 -e MatrixMultiplication/Dataset/1/output.raw -i MatrixMultiplication/Dataset/1/input0.raw,MatrixMultiplication/Dataset/1/input1.raw  -t matrix

run2:
	./ex1 -e MatrixMultiplication/Dataset/2/output.raw -i MatrixMultiplication/Dataset/2/input0.raw,MatrixMultiplication/Dataset/2/input1.raw  -t matrix

run3:
	./ex1 -e MatrixMultiplication/Dataset/3/output.raw -i MatrixMultiplication/Dataset/3/input0.raw,MatrixMultiplication/Dataset/3/input1.raw  -t matrix

run4:
	./ex1 -e MatrixMultiplication/Dataset/4/output.raw -i MatrixMultiplication/Dataset/4/input0.raw,MatrixMultiplication/Dataset/4/input1.raw  -t matrix

run5:
	./ex1 -e MatrixMultiplication/Dataset/5/output.raw -i MatrixMultiplication/Dataset/5/input0.raw,MatrixMultiplication/Dataset/5/input1.raw  -t matrix

PROFILE=/usr/local/cuda-11.8/bin/ncu --export ./Example1_profile \
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
	$(PROFILE) ./ex1 -e MatrixMultiplication/Dataset/0/output.raw -i MatrixMultiplication/Dataset/0/input0.raw,MatrixMultiplication/Dataset/0/input1.raw -t matrix

prof1:	
	$(PROFILE) ./ex1 -e MatrixMultiplication/Dataset/1/output.raw -i MatrixMultiplication/Dataset/1/input0.raw,MatrixMultiplication/Dataset/1/input1.raw  -t matrix

prof2:	
	$(PROFILE) ./ex1 -e MatrixMultiplication/Dataset/2/output.raw -i MatrixMultiplication/Dataset/2/input0.raw,MatrixMultiplication/Dataset/2/input1.raw  -t matrix

prof3:	
	$(PROFILE) ./ex1 -e MatrixMultiplication/Dataset/3/output.raw -i MatrixMultiplication/Dataset/3/input0.raw,MatrixMultiplication/Dataset/3/input1.raw  -t matrix

prof4:	
	$(PROFILE) ./ex1 -e MatrixMultiplication/Dataset/4/output.raw -i MatrixMultiplication/Dataset/4/input0.raw,MatrixMultiplication/Dataset/4/input1.raw  -t matrix

prof5:	
	$(PROFILE) ./ex1 -e MatrixMultiplication/Dataset/5/output.raw -i MatrixMultiplication/Dataset/5/input0.raw,MatrixMultiplication/Dataset/5/input1.raw  -t matrix



clean:
	rm -f ex1 generator
