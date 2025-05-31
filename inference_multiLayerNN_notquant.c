#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

// Configuration
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define HIDDEN_SIZE 256
#define NUM_TEST_IMAGES 10000
#define MAX_LAYERS 8
#define MAX_NEURONS 1024
#define FIXED_POINT 1
#define ALIGNMENT 64
#define FIXED_SCALE (1LL << 20)
#define NUM_THREADS 4

// Fixed-point arithmetic
#if FIXED_POINT
typedef int32_t fixed_t;
#define FLOAT_TO_FIXED(x) ((fixed_t)((x) * FIXED_SCALE))
#define FIXED_TO_FLOAT(x) ((float)(x) / FIXED_SCALE)
#define FIXED_MUL(x, y) (((int64_t)(x) * (y)) / FIXED_SCALE)
#define FIXED_DIV(x, y) (((int64_t)(x) * FIXED_SCALE) / (y))
#else
typedef float fixed_t;
#define FLOAT_TO_FIXED(x) (x)
#define FIXED_TO_FLOAT(x) (x)
#define FIXED_MUL(x, y) ((x) * (y))
#define FIXED_DIV(x, y) ((x) / (y))
#endif

// Logging
#define LOG(fmt, ...) fprintf(stderr, "[INFO] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define ERR(fmt, ...) fprintf(stderr, "[ERROR] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

// Data structure
typedef struct {
    float *training_images;
    int *training_labels;
    float *test_images;
    int *test_labels;
} data;

// Layer structure
typedef struct {
    int size;
    fixed_t *activations;
    fixed_t *weights;
    fixed_t *biases;
} layer;

// Neural network structure
typedef struct {
    int n; // Number of hidden layers
    layer *layers;
    int num_correct_predictions;
} NN;

// Memory pool
struct MemoryPool {
    uint8_t *buffer;
    size_t size;
    size_t offset;
};

struct MemoryPool *create_memory_pool(size_t size) {
    struct MemoryPool *pool = malloc(sizeof(struct MemoryPool));
    if (!pool) {
        ERR("Memory pool allocation failed");
        exit(1);
    }
    pool->buffer = aligned_alloc(ALIGNMENT, size);
    if (!pool->buffer) {
        ERR("Memory pool buffer allocation failed");
        free(pool);
        exit(1);
    }
    pool->size = size;
    pool->offset = 0;
    return pool;
}

void *pool_alloc(struct MemoryPool *pool, size_t size) {
    size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    if (pool->offset + aligned_size > pool->size) {
        ERR("Memory pool out of space: %zu requested", aligned_size);
        exit(1);
    }
    void *ptr = pool->buffer + pool->offset;
    pool->offset += aligned_size;
    return ptr;
}

void free_memory_pool(struct MemoryPool *pool) {
    free(pool->buffer);
    free(pool);
}

// Utility functions
int reverse_int(int i) {
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist_images(const char *filename, float *images, int num_images) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        ERR("Could not open file %s", filename);
        exit(1);
    }
    int magic_number, number_of_images, rows, cols;
    if (fread(&magic_number, sizeof(int), 1, fp) != 1) {
        ERR("Failed to read magic number");
        fclose(fp);
        exit(1);
    }
    magic_number = reverse_int(magic_number);
    if (fread(&number_of_images, sizeof(int), 1, fp) != 1) {
        ERR("Failed to read number of images");
        fclose(fp);
        exit(1);
    }
    number_of_images = reverse_int(number_of_images);
    if (fread(&rows, sizeof(int), 1, fp) != 1 || fread(&cols, sizeof(int), 1, fp) != 1) {
        ERR("Failed to read dimensions");
        fclose(fp);
        exit(1);
    }
    rows = reverse_int(rows);
    cols = reverse_int(cols);
    for (int i = 0; i < num_images; ++i) {
        for (int r = 0; r < rows * cols; ++r) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, fp) != 1) {
                ERR("Failed to read pixel at image %d, pos %d", i, r);
                fclose(fp);
                exit(1);
            }
            images[i * INPUT_SIZE + r] = pixel / 255.0f;
        }
    }
    fclose(fp);
}

void read_mnist_labels(const char *filename, int *labels, int num_labels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        ERR("Could not open file %s", filename);
        exit(1);
    }
    int magic_number, number_of_labels;
    if (fread(&magic_number, sizeof(int), 1, fp) != 1 || fread(&number_of_labels, sizeof(int), 1, fp) != 1) {
        ERR("Failed to read label headers");
        fclose(fp);
        exit(1);
    }
    magic_number = reverse_int(magic_number);
    number_of_labels = reverse_int(number_of_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, fp) != 1) {
            ERR("Failed to read label %d", i);
            fclose(fp);
            exit(1);
        }
        labels[i] = (int)label;
    }
    fclose(fp);
}

void malloc_data(data *dataset) {
    dataset->test_images = malloc(NUM_TEST_IMAGES * INPUT_SIZE * sizeof(float));
    dataset->test_labels = malloc(NUM_TEST_IMAGES * sizeof(int));
    dataset->training_images = malloc(NUM_TEST_IMAGES * INPUT_SIZE * sizeof(float)); // Minimal allocation
    dataset->training_labels = malloc(NUM_TEST_IMAGES * sizeof(int));
    if (!dataset->test_images || !dataset->test_labels || !dataset->training_images || !dataset->training_labels) {
        ERR("Data allocation failed");
        exit(1);
    }
}

void free_data(data *dataset) {
    free(dataset->test_images);
    free(dataset->test_labels);
    free(dataset->training_images);
    free(dataset->training_labels);
}

// Activation functions
void relu(fixed_t *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

void softmax(fixed_t *x, int size, fixed_t *output) {
    int64_t max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) max = x[i];
    }
    int64_t sum = 0;
    for (int i = 0; i < size; i++) {
        float val = FIXED_TO_FLOAT(x[i] - max);
        if (val < -20.0f) val = -20.0f;
        if (val > 20.0f) val = 20.0f;
        output[i] = FLOAT_TO_FIXED(expf(val));
        sum += output[i];
    }
    if (sum == 0) sum = 1;
    for (int i = 0; i < size; i++) {
        output[i] = FIXED_DIV(output[i], sum);
    }
}

int max_index(fixed_t *out, int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++) {
        if (out[i] > out[max_i]) max_i = i;
    }
    return max_i;
}

// Network setup
void set_sizes(NN *rs) {
    rs->layers[0].size = INPUT_SIZE;
    for (int i = 0; i < rs->n; i++) {
        rs->layers[i + 1].size = HIDDEN_SIZE;
    }
    rs->layers[rs->n + 1].size = OUTPUT_SIZE;
}

void alloc_runstate(struct MemoryPool *pool, NN *rs, int n) {
    rs->n = n;
    rs->layers = pool_alloc(pool, (n + 2) * sizeof(layer));
    set_sizes(rs);
    for (int i = 0; i < n + 1; i++) {
        rs->layers[i].activations = pool_alloc(pool, rs->layers[i].size * sizeof(fixed_t));
        rs->layers[i].weights = pool_alloc(pool, rs->layers[i + 1].size * rs->layers[i].size * sizeof(fixed_t));
        rs->layers[i].biases = pool_alloc(pool, rs->layers[i + 1].size * sizeof(fixed_t));
    }
    rs->layers[n + 1].activations = pool_alloc(pool, OUTPUT_SIZE * sizeof(fixed_t));
    rs->layers[n + 1].weights = NULL;
    rs->layers[n + 1].biases = NULL;
}

void load_weights(NN *model, const char *file_name) {
    FILE *file = fopen(file_name, "rb");
    if (!file) {
        ERR("Error opening file %s", file_name);
        exit(1);
    }
    int n;
    if (fread(&n, sizeof(int), 1, file) != 1) {
        ERR("Failed to read number of hidden layers");
        fclose(file);
        exit(1);
    }
    struct MemoryPool *pool = create_memory_pool((n + 2) * MAX_NEURONS * (MAX_NEURONS + 1) * sizeof(fixed_t));
    alloc_runstate(pool, model, n);
    for (int i = 0; i < model->n + 1; i++) {
        float *temp_weights = malloc(model->layers[i].size * model->layers[i + 1].size * sizeof(float));
        float *temp_biases = malloc(model->layers[i + 1].size * sizeof(float));
        if (!temp_weights || !temp_biases) {
            ERR("Temporary buffer allocation failed");
            free(temp_weights);
            free(temp_biases);
            fclose(file);
            exit(1);
        }
        size_t w_size = model->layers[i].size * model->layers[i + 1].size;
        if (fread(temp_weights, sizeof(float), w_size, file) != w_size) {
            ERR("Error reading weights for layer %d", i);
            free(temp_weights);
            free(temp_biases);
            fclose(file);
            exit(1);
        }
        if (fread(temp_biases, sizeof(float), model->layers[i + 1].size, file) != model->layers[i + 1].size) {
            ERR("Error reading biases for layer %d", i);
            free(temp_weights);
            free(temp_biases);
            fclose(file);
            exit(1);
        }
        for (size_t j = 0; j < w_size; j++) {
            model->layers[i].weights[j] = FLOAT_TO_FIXED(temp_weights[j]);
        }
        for (size_t j = 0; j < model->layers[i + 1].size; j++) {
            model->layers[i].biases[j] = FLOAT_TO_FIXED(temp_biases[j]);
        }
        free(temp_weights);
        free(temp_biases);
    }
    fclose(file);
    LOG("Loaded weights from %s", file_name);
}

// Forward function with three implementations
#if defined(USE_OPENMP)
void forward(NN *model, int layer_number) {
    int d = model->layers[layer_number + 1].size;
    int n = model->layers[layer_number].size;
    fixed_t *input = model->layers[layer_number].activations;
    fixed_t *output = model->layers[layer_number + 1].activations;
    fixed_t *weights = model->layers[layer_number].weights;
    fixed_t *biases = model->layers[layer_number].biases;

    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for (int i = 0; i < d; i++) {
        int64_t sum = (int64_t)biases[i] * FIXED_SCALE;
        for (int j = 0; j < n; j++) {
            sum += (int64_t)weights[i * n + j] * input[j];
        }
        output[i] = (fixed_t)(sum / FIXED_SCALE);
        if (output[i] > FLOAT_TO_FIXED(100.0f)) output[i] = FLOAT_TO_FIXED(100.0f);
        if (output[i] < FLOAT_TO_FIXED(-100.0f)) output[i] = FLOAT_TO_FIXED(-100.0f);
    }
}
#elif defined(USE_PTHREADS)
typedef struct {
    int start_idx;
    int end_idx;
    int input_size;
    fixed_t *weights;
    fixed_t *biases;
    fixed_t *input;
    fixed_t *output;
} NeuronTask;

void *compute_neurons(void *arg) {
    NeuronTask *task = (NeuronTask *)arg;
    for (int i = task->start_idx; i < task->end_idx; i++) {
        int64_t sum = (int64_t)task->biases[i] * FIXED_SCALE;
        for (int j = 0; j < task->input_size; j++) {
            sum += (int64_t)task->weights[i * task->input_size + j] * task->input[j];
        }
        task->output[i] = (fixed_t)(sum / FIXED_SCALE);
        if (task->output[i] > FLOAT_TO_FIXED(100.0f)) task->output[i] = FLOAT_TO_FIXED(100.0f);
        if (task->output[i] < FLOAT_TO_FIXED(-100.0f)) task->output[i] = FLOAT_TO_FIXED(-100.0f);
    }
    return NULL;
}

void forward(NN *model, int layer_number) {
    int d = model->layers[layer_number + 1].size;
    int n = model->layers[layer_number].size;
    fixed_t *input = model->layers[layer_number].activations;
    fixed_t *output = model->layers[layer_number + 1].activations;
    fixed_t *weights = model->layers[layer_number].weights;
    fixed_t *biases = model->layers[layer_number].biases;

    int chunk_size = (d + NUM_THREADS - 1) / NUM_THREADS;
    pthread_t threads[NUM_THREADS];
    NeuronTask tasks[NUM_THREADS];
    int active_threads = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        int start = t * chunk_size;
        int end = (t + 1) * chunk_size < d ? (t + 1) * chunk_size : d;
        if (start >= d) break;
        tasks[t] = (NeuronTask){start, end, n, weights, biases, input, output};
        if (pthread_create(&threads[t], NULL, compute_neurons, &tasks[t]) != 0) {
            ERR("Failed to create thread %d", t);
            exit(1);
        }
        active_threads++;
    }
    for (int t = 0; t < active_threads; t++) {
        if (pthread_join(threads[t], NULL) != 0) {
            ERR("Failed to join thread %d", t);
            exit(1);
        }
    }
}
#else
void forward(NN *model, int layer_number) {
    int d = model->layers[layer_number + 1].size;
    int n = model->layers[layer_number].size;
    fixed_t *input = model->layers[layer_number].activations;
    fixed_t *output = model->layers[layer_number + 1].activations;
    fixed_t *weights = model->layers[layer_number].weights;
    fixed_t *biases = model->layers[layer_number].biases;

    for (int i = 0; i < d; i++) {
        int64_t sum = (int64_t)biases[i] * FIXED_SCALE;
        for (int j = 0; j < n; j++) {
            sum += (int64_t)weights[i * n + j] * input[j];
        }
        output[i] = (fixed_t)(sum / FIXED_SCALE);
        if (output[i] > FLOAT_TO_FIXED(100.0f)) output[i] = FLOAT_TO_FIXED(100.0f);
        if (output[i] < FLOAT_TO_FIXED(-100.0f)) output[i] = FLOAT_TO_FIXED(-100.0f);
    }
}
#endif

void prep_rs(NN *rs, data *dataset, int image_number) {
    int offset = image_number * INPUT_SIZE;
    for (int i = 0; i < INPUT_SIZE; i++) {
        rs->layers[0].activations[i] = FLOAT_TO_FIXED(dataset->test_images[offset + i]);
    }
}

void test(NN *model, int correct_label) {
    int out_layer_number = model->n + 1;
    for (int i = 0; i < model->n; i++) {
        forward(model, i);
        relu(model->layers[i + 1].activations, model->layers[i + 1].size);
    }
    forward(model, model->n);
    fixed_t *output = model->layers[out_layer_number].activations;
    fixed_t temp[OUTPUT_SIZE];
    softmax(output, OUTPUT_SIZE, temp);
    memcpy(output, temp, OUTPUT_SIZE * sizeof(fixed_t));
    int index = max_index(output, OUTPUT_SIZE);
    if (index == correct_label) {
        model->num_correct_predictions++;
    }
}

int main() {
    char *file_name = "saved_model.NN";
    NN model_1 = {0};
    data dataset_1 = {0};
    malloc_data(&dataset_1);
    LOG("Loading test data...");
    read_mnist_images("./data/t10k-images.idx3-ubyte", dataset_1.test_images, NUM_TEST_IMAGES);
    read_mnist_labels("./data/t10k-labels.idx1-ubyte", dataset_1.test_labels, NUM_TEST_IMAGES);
    LOG("Loaded test data");
    load_weights(&model_1, file_name);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    model_1.num_correct_predictions = 0;
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        prep_rs(&model_1, &dataset_1, i);
        test(&model_1, dataset_1.test_labels[i]);
        if ((i + 1) % 1000 == 0) {
            LOG("%d images processed... accuracy: %f", i + 1, 
                (float)model_1.num_correct_predictions / (i + 1));
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    long ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;

    float accuracy = (float)model_1.num_correct_predictions / NUM_TEST_IMAGES;
    printf("Test Accuracy: %.4f\n", accuracy);
    printf("Inference Time: %ld ms\n", ms);

    // free_data(&dataset_1);
    // free_memory_pool((struct MemoryPool *)model_1.layers[0].activations); // Pool allocated in load_weights
    return 0;
}