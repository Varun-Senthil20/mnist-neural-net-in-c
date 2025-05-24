#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
const int hidden_size=256;
const int input_size=784;
const int output_size=10;
const int num_train_images=60000;//might need to change these num_..._images things when we switch datasets.
const int num_test_images=10000;
const int num_epochs=10;

typedef struct{
float *training_images;//num_train_images*input_size
int *training_labels;
float *test_images;
int *test_labels;
}data;
typedef struct{
int size;
float *activations;
float *weights;
float *biases;
} layer;
typedef struct{
    int n;
    layer *layers;
    int num_correct_predictions;
}NN;
int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
// Read MNIST images
void read_mnist_images(const char *filename, float*images, int num_images) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Could not open file %s\n", filename);
        exit(1);
    }
    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = reverse_int(magic_number);

    fread(&number_of_images, sizeof(int), 1, fp);
    number_of_images = reverse_int(number_of_images);

    fread(&rows, sizeof(int), 1, fp);
    rows = reverse_int(rows);

    fread(&cols, sizeof(int), 1, fp);
    cols = reverse_int(cols);

    for (int i = 0; i < num_images; ++i) {
        for (int r = 0; r < rows * cols; ++r) {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(unsigned char), 1, fp);
            images[i*input_size+r] = (float)(pixel / 255.0); // Normalize pixel values
        }
    }
    fclose(fp);
}

// Read MNIST labels
void read_mnist_labels(const char *filename, int *labels, int num_labels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Could not open file %s\n", filename);
        exit(1);
    }
    int magic_number = 0;
    int number_of_labels = 0;

    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = reverse_int(magic_number);

    fread(&number_of_labels, sizeof(int), 1, fp);
    number_of_labels = reverse_int(number_of_labels);

    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        fread(&label, sizeof(unsigned char), 1, fp);
        labels[i] = (int)label;
    }
    fclose(fp);
}
void set_sizes(NN *rs)
{
    rs->layers[0].size=input_size;
    for(int i=0;i<rs->n;i++)
    {
        rs->layers[i+1].size=256;//can get each layer size as an input here or make a config file format which i can read here, future work....
    }
    rs->layers[rs->n+1].size=output_size;
}
void alloc_runstate(NN *rs,int n)
{
    rs->layers=(layer*)malloc((rs->n+2)*sizeof(layer));
    set_sizes(rs);
    for(int i=0;i<rs->n+1;i++)//takes care of all layers except the output layer;
    {
        rs->layers[i].activations=(float*)calloc(rs->layers[i].size,sizeof(float));
        rs->layers[i].biases=(float*)calloc(rs->layers[i+1].size,sizeof(float));
        rs->layers[i].weights=(float*)calloc(rs->layers[i+1].size*rs->layers[i].size,sizeof(float));
    }
    //taking care of the output layer
    rs->layers[rs->n+1].activations=(float*)calloc(output_size,sizeof(float));
    rs->layers[rs->n+1].biases=NULL;
    rs->layers[rs->n+1].weights=NULL;
}
void free_runstate(NN *rs)
{
    for(int i=0;i<rs->n+1;i++)//takes care of all layers except the output layer;
    {
        free(rs->layers[i].activations);
        free(rs->layers[i].biases);
        free(rs->layers[i].weights);
    }
}
void free_data(data *dataset)
{
    free(dataset->test_images);
    free(dataset->test_labels);
    free(dataset->training_images);
    free(dataset->training_labels);
}
void malloc_data(data* dataset)
{
    dataset->test_images=(float*)malloc(num_test_images*input_size*sizeof(float));
    dataset->test_labels=(int*)malloc(num_test_images*sizeof(int));
    dataset->training_images=(float*)malloc(num_train_images*input_size*sizeof(float));
    dataset->training_labels=(int*)malloc(num_train_images*sizeof(int));
}
void prep_rs(NN *rs, data *dataset,int image_number)
{
    int previous_images=image_number*input_size;
    for(int i=0;i<input_size;i++)
    {
        rs->layers[0].activations[i]=dataset->test_images[previous_images+i];
    }

}
void sigmoid(float *x, int n)
{
    for(int i=0;i<n;i++)
    {
        x[i]=1.0 / (1.0 + exp(-x[i]));
    }
}
void relu(float *x, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        if(x[i]<0){
            x[i]=0;
        }
    }

}
void softmax(float *x, int size) {
    // softmax as defined here: https://en.wikipedia.org/wiki/Softmax_function

    // Find the maximum value across all input values
    float max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) max = x[i];
    }

    // Compute the exponentials of the input values
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }

    // Normalize the output values
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}
int max_index(float *out, int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++) {
        if (out[i] > out[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}
void forward(NN *model ,int layer_number) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    //#pragma omp parallel for private(i)
    int d=model->layers[layer_number+1].size;
    int n=model->layers[layer_number].size;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += model->layers[layer_number].weights[i * n + j] * model->layers[layer_number].activations[j];
        }
        model->layers[layer_number+1].activations[i] = val+model->layers[layer_number].biases[i];
    }
}
void matmul(float* xout, float* x, float* w, float*b ,int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val+b[i];
    }
}
void load_weights(NN *model, char* file_name)
{
    FILE* file = fopen(file_name, "rb");
    int n;
    if (file == NULL) {
        printf("Error opening file\n");
        exit(1);
    }
    fread(&n, sizeof(int),1, file);
    printf("This model has %d layers\n",n);
    model->n=n;
    alloc_runstate(model,n);
    for(int i=0;i<model->n+1;i++)//takes care of all layers except the output layer;
    {
        fread(model->layers[i].weights, sizeof(float), model->layers[i].size * model->layers[i+1].size, file);
        fread(model->layers[i].biases, sizeof(float), model->layers[i+1].size, file);
    }
    fclose(file);
}
void test(NN *model, int correct_label)
{
    float learning_rate = 0.01;
    int i,j;
    int out_layer_number=model->n+1;
    // Feedforward
    for(i=0;i<model->n;i++)
    {
        forward(model,i);
        relu(model->layers[i+1].activations,model->layers[i+1].size);
    }
    forward(model,i);
    //sigmoid(rs->out,output_size);
    softmax(model->layers[out_layer_number].activations,output_size);

    int index = max_index(model->layers[out_layer_number].activations, output_size);

    if (index == correct_label) {
        model->num_correct_predictions++;
    }
}
int main()
{
    char *file_name = "saved_model.NN";
    int n=1;//number of hidden layers, might be able to set up a file format where even this is read, future work ...
    NN model_1;
    data dataset_1;
    malloc_data(&dataset_1);
    // Read training data
    printf("Loading test data...\n");
    read_mnist_images("./data/t10k-images.idx3-ubyte", dataset_1.test_images, num_test_images);
    read_mnist_labels("./data/t10k-labels.idx1-ubyte", dataset_1.test_labels, num_test_images);
    printf("Loaded test data\n");
    load_weights(&model_1,file_name);
    //random_weights(&model_1);//comment this and uncomment the previous line after try_1 to make sure that load works.
    model_1.num_correct_predictions=0;
    for (int i = 1; i <=num_test_images; i++)
    {
        //need to updte rs->inp  and correct_label here
        prep_rs(&model_1,&dataset_1,i-1);
        //rs_1.inp=(float*)(dataset_1.training_images+i*input_size*sizeof(float));
        test(&model_1,dataset_1.test_labels[i-1]);
        if(i%1000==0)
        {
            printf("%d images processed... accuracy: %f\n",i,(float)model_1.num_correct_predictions / i);
            fflush(stdout);
        }
    }
    printf("Test Accuracy: %f\n",(float) model_1.num_correct_predictions / num_test_images);
    fflush(stdout);
    
    free_runstate(&model_1);
    free_data(&dataset_1);
    return 0;
}
