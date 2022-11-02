
all:
#	nvcc solution.cu -O3 -g -G -o ex2 -I../libwb/ ../libwb/lib/libwb.a
	nvcc template.cu -O3 -g -G -o ex2 -I../libwb/ ../libwb/lib/libwb.a
	g++ dataset_generator.cpp -O3 -o generator -I../libwb/ ../libwb/lib/libwb.a

gen:
	./generator

run0:
	./ex2 -e VectorAdd/Dataset/0/output.raw -i VectorAdd/Dataset/0/input0.raw,VectorAdd/Dataset/0/input1.raw -t vector
	
run1:
	./ex2 -e VectorAdd/Dataset/1/output.raw -i VectorAdd/Dataset/1/input0.raw,VectorAdd/Dataset/1/input1.raw -t vector
	
run2:
	./ex2 -e VectorAdd/Dataset/2/output.raw -i VectorAdd/Dataset/2/input0.raw,VectorAdd/Dataset/2/input1.raw -t vector

run3:
	./ex2 -e VectorAdd/Dataset/3/output.raw -i VectorAdd/Dataset/3/input0.raw,VectorAdd/Dataset/3/input1.raw -t vector
	
run4:
	./ex2 -e VectorAdd/Dataset/4/output.raw -i VectorAdd/Dataset/4/input0.raw,VectorAdd/Dataset/4/input1.raw -t vector

run5:
	./ex2 -e VectorAdd/Dataset/5/output.raw -i VectorAdd/Dataset/5/input0.raw,VectorAdd/Dataset/5/input1.raw -t vector
	
run6:
	./ex2 -e VectorAdd/Dataset/6/output.raw -i VectorAdd/Dataset/6/input0.raw,VectorAdd/Dataset/6/input1.raw -t vector


clean:
	rm -f ex2 generator
