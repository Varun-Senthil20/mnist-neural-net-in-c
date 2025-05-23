
run: inference_multiLayerNN_notquant.c  multiLayerNN_notquant.c
	gcc -O3 -march=native -funroll-loops -o train.o multiLayerNN_notquant.c -fopenmp -lopenblas -lm
	gcc -o test.o inference_multiLayerNN_notquant.c -lm

data_download:
	bash data_download.sh
