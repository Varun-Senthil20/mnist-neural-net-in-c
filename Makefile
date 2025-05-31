
run_omp: inference_multiLayerNN_notquant.c  multiLayerNN_notquant.c
	gcc -O3 -march=native -funroll-loops -o train.o multiLayerNN_notquant.c -fopenmp -lopenblas -lm
	gcc -o test.o inference_multiLayerNN_notquant.c -lm -cpp -fopenmp -DUSE_OPENMP
	./test.o
run_pthread: inference_multiLayerNN_notquant.c  multiLayerNN_notquant.c
	gcc -O3 -march=native -funroll-loops -o train.o multiLayerNN_notquant.c -pthread -lopenblas -lm
	gcc -o test.o inference_multiLayerNN_notquant.c -lm -cpp -pthread -DUSE_PTHREADS
	./test.o
run_serial: inference_multiLayerNN_notquant.c  multiLayerNN_notquant.c
	gcc -O3 -march=native -funroll-loops -o train.o multiLayerNN_notquant.c -lopenblas -lm
	gcc -o test.o inference_multiLayerNN_notquant.c -lm -cpp
	./test.o
data_download:
	bash data_download.sh
