#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// Define number types
#define TYPE_INTEGER 1
#define TYPE_FLOAT 2
#define TYPE_MIXED 3

// Structure for thread arguments
typedef struct {
    float **A;
    float **B;
    float **C;
    int m;      
    int n;      
    int p;      
    int row;    
} ThreadArgs;

// Function prototypes
void *multiply_row(void *args);
void multiply_single_thread(float **A, float **B, float **C, int m, int n, int p);
void multiply_multi_thread(float **A, float **B, float **C, int m, int n, int p);
float **create_matrix(int rows, int cols);
void free_matrix(float **matrix, int rows);
void print_matrix(float **matrix, int rows, int cols, int num_type);
float **read_matrix_from_file(int rows, int cols, const char *filename);
float **generate_random_matrix(int rows, int cols, int num_type, float min_val, float max_val);
void clear_input_buffer();
void print_separator();

// Clear input buffer function
void clear_input_buffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

// Print separator line
void print_separator() {
    printf("\n----------------------------------------\n");
}

// Thread function to multiply one row
void *multiply_row(void *args) {
    ThreadArgs *thread_args = (ThreadArgs *)args;
    int row = thread_args->row;
    
    for (int k = 0; k < thread_args->p; k++) {
        thread_args->C[row][k] = 0.0;
        for (int j = 0; j < thread_args->n; j++) {
            thread_args->C[row][k] += thread_args->A[row][j] * thread_args->B[j][k];
        }
    }
    pthread_exit(NULL);
}

// Single-threaded matrix multiplication
void multiply_single_thread(float **A, float **B, float **C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < p; k++) {
            C[i][k] = 0.0;
            for (int j = 0; j < n; j++) {
                C[i][k] += A[i][j] * B[j][k];
            }
        }
    }
}

// Multi-threaded matrix multiplication
void multiply_multi_thread(float **A, float **B, float **C, int m, int n, int p) {
    pthread_t *threads = malloc(m * sizeof(pthread_t));
    ThreadArgs *thread_args = malloc(m * sizeof(ThreadArgs));

    for (int i = 0; i < m; i++) {
        thread_args[i].A = A;
        thread_args[i].B = B;
        thread_args[i].C = C;
        thread_args[i].m = m;
        thread_args[i].n = n;
        thread_args[i].p = p;
        thread_args[i].row = i;
        
        if (pthread_create(&threads[i], NULL, multiply_row, &thread_args[i]) != 0) {
            printf("Error creating thread %d\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < m; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_args);
}

// Create matrix
float **create_matrix(int rows, int cols) {
    float **matrix = malloc(rows * sizeof(float *));
    if (matrix == NULL) {
        printf("Error: Memory allocation failed\n");
        exit(1);
    }
    
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(float));
        if (matrix[i] == NULL) {
            printf("Error: Memory allocation failed\n");
            exit(1);
        }
    }
    return matrix;
}

// Print matrix with appropriate formatting based on number type
void print_matrix(float **matrix, int rows, int cols, int num_type) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (num_type == TYPE_INTEGER) {
                printf("%4d ", (int)matrix[i][j]);
            } else {
                printf("%7.3f ", matrix[i][j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

// Generate random matrix with specified number type and range
float **generate_random_matrix(int rows, int cols, int num_type, float min_val, float max_val) {
    float **matrix = create_matrix(rows, cols);
    float range = max_val - min_val;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (num_type == TYPE_INTEGER) {
                matrix[i][j] = (int)(min_val + (rand() % (int)(range + 1)));
            } 
            else if (num_type == TYPE_FLOAT) {
                matrix[i][j] = min_val + ((float)rand() / RAND_MAX) * range;
            }
            else { // TYPE_MIXED
                if (rand() % 2) {
                    matrix[i][j] = (int)(min_val + (rand() % (int)(range + 1)));
                } else {
                    matrix[i][j] = min_val + ((float)rand() / RAND_MAX) * range;
                }
            }
        }
    }
    return matrix;
}

// Read matrix from file
float **read_matrix_from_file(int rows, int cols, const char *filename) {
    float **matrix = create_matrix(rows, cols);
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%f", &matrix[i][j]) != 1) {
                printf("Error: Reading from file %s failed\n", filename);
                exit(1);
            }
        }
    }
    fclose(file);
    return matrix;
}

// Free matrix memory
void free_matrix(float **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(int argc, char *argv[]) {
    print_separator();
    printf("    Matrix Multiplication Program\n");
    print_separator();
    
    // Check command line arguments
    // Check command line arguments
    if (argc != 5) {  // Changed from 4 to 5
        printf("Usage: %s <rows_A> <cols_A> <rows_B> <cols_B>\n", argv[0]);
        printf("Example: %s 3 4 4 2\n", argv[0]);
        return 1;
    }

    // Get and validate matrix dimensions
    int rows_A = atoi(argv[1]);
    int cols_A = atoi(argv[2]);
    int rows_B = atoi(argv[3]);
    int cols_B = atoi(argv[4]);

    // Validate positive dimensions
    if (rows_A <= 0 || cols_A <= 0 || rows_B <= 0 || cols_B <= 0) {
        printf("Error: Matrix dimensions must be positive numbers\n");
        return 1;
    }

    // Check multiplication compatibility
    if (cols_A != rows_B) {
        printf("Error: Matrices cannot be multiplied!\n");
        printf("Number of columns in Matrix A (%d) must equal number of rows in Matrix B (%d)\n", 
               cols_A, rows_B);
        return 1;
    }

    // Store dimensions for multiplication
    int m = rows_A;
    int n = cols_A;  // same as rows_B
    int p = cols_B;

    printf("\nMatrix Dimensions:\n");
    printf("Matrix A: %d x %d\n", rows_A, cols_A);
    printf("Matrix B: %d x %d\n", rows_B, cols_B);
    print_separator();

    // Initialize random seed
    srand(time(NULL));

    float **A = NULL, **B = NULL, **C_single = NULL, **C_multi = NULL;
    
    // Input method selection
    int choice;
    printf("\nChoose input method:\n");
    printf("1. Read matrices from files\n");
    printf("2. Generate random matrices\n");
    printf("Enter choice (1 or 2): ");
    scanf("%d", &choice);
    clear_input_buffer();

    int num_type = TYPE_FLOAT;
    float min_val = 0.0, max_val = 10.0;

    if (choice == 1) {
        char filename[100];
        printf("\nEnter filename for matrix A: ");
        scanf("%s", filename);
        A = read_matrix_from_file(m, n, filename);
        
        printf("Enter filename for matrix B: ");
        scanf("%s", filename);
        B = read_matrix_from_file(n, p, filename);
    } 
    else if (choice == 2) {
        print_separator();
        printf("Choose number type for random generation:\n");
        printf("1. Integers only\n");
        printf("2. Floating-point numbers\n");
        printf("3. Mixed (both integers and floating-point)\n");
        printf("Enter choice (1-3): ");
        scanf("%d", &num_type);
        clear_input_buffer();

        printf("\nEnter range for random numbers:\n");
        printf("Minimum value: ");
        scanf("%f", &min_val);
        printf("Maximum value: ");
        scanf("%f", &max_val);
        clear_input_buffer();

        printf("\nGenerating random matrices...\n");
        A = generate_random_matrix(m, n, num_type, min_val, max_val);
        B = generate_random_matrix(n, p, num_type, min_val, max_val);
    }
    else {
        printf("Invalid choice\n");
        return 1;
    }

    C_single = create_matrix(m, p);
    C_multi = create_matrix(m, p);

    // Perform calculations and measure time
    // Perform calculations and measure time with multiple iterations
    struct timeval start, end;
    const int NUM_ITERATIONS = 10;  // Increased number of iterations for better accuracy
    double time_single = 0.0;
    double time_multi = 0.0;
    
    print_separator();
    printf("Calculating (Running %d iterations for accurate timing)...\n", NUM_ITERATIONS);

    // Warm up run to stabilize CPU frequency and cache
    multiply_single_thread(A, B, C_single, m, n, p);
    multiply_multi_thread(A, B, C_multi, m, n, p);
    
    // Multiple iterations for single thread
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        gettimeofday(&start, NULL);
        multiply_single_thread(A, B, C_single, m, n, p);
        gettimeofday(&end, NULL);
        time_single += (end.tv_sec - start.tv_sec) + 
                      (end.tv_usec - start.tv_usec) / 1000000.0;
    }
    time_single /= NUM_ITERATIONS;

    // Multiple iterations for multi thread
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        gettimeofday(&start, NULL);
        multiply_multi_thread(A, B, C_multi, m, n, p);
        gettimeofday(&end, NULL);
        time_multi += (end.tv_sec - start.tv_sec) + 
                     (end.tv_usec - start.tv_usec) / 1000000.0;
    }
    time_multi /= NUM_ITERATIONS;

    // Print results
    print_separator();
    printf("Results\n");
    print_separator();
    
    printf("\nMatrix A:\n");
    print_matrix(A, m, n, num_type);
    
    printf("Matrix B:\n");
    print_matrix(B, n, p, num_type);
    
    printf("Result (Single-threaded):\n");
    print_matrix(C_single, m, p, num_type);
    
    printf("Result (Multi-threaded):\n");
    print_matrix(C_multi, m, p, num_type);

    // Print performance comparison
    print_separator();
    printf("Performance Comparison\n");
    print_separator();
    printf("Single-threaded time: %.9f seconds (averaged over %d runs)\n", 
           time_single, NUM_ITERATIONS);
    printf("Multi-threaded time:  %.9f seconds (averaged over %d runs)\n", 
           time_multi, NUM_ITERATIONS);
    
    
    if (time_multi > 0) {
        printf("Speedup Formula = (Single-threaded time / Multi-threaded time)\n");
        printf("              = (%.9f / %.9f)\n", time_single, time_multi);
        printf("Speedup: %.3fx\n", time_single/time_multi);
        printf("Performance improvement: %.2f%%\n", ((time_single/time_multi) - 1) * 100);
    }

    // Free memory
    free_matrix(A, m);
    free_matrix(B, n);
    free_matrix(C_single, m);
    free_matrix(C_multi, m);

    return 0;

}