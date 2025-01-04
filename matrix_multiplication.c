#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h> // for random number seeding
#include <sys/time.h> // to measure time
#include <math.h> // for pow function


// number types use in matrices
#define TYPE_INTEGER 1
#define TYPE_FLOAT 2
#define TYPE_MIXED 3

// structure to hold necessary information for each thread
typedef struct {
    float **matrixA; // input matrix A
    float **matrixB; // input matrix B
    float **outputMatrix; // output matrix
    int rowsA;      // number of rows in matrix A
    int columnsA;      // number of columns in matrix A (same as rows in matrix B)
    int columnsB;      // number of columns in matrix B
    int row;    // row number to calculate
} ThreadParameters;

void *calculate_matrix_row(void *args);
void multiply_single_thread(float **matrixA, float **matrixB, float **outputMatrix, int m, int n, int p);
void multiply_multiple_threads(float **matrixA, float **matrixB, float **outputMatrix, int m, int n, int p);
float **create_matrix(int rows, int cols);
void free_matrix_memory(float **matrix, int rows);
void print_matrix(float **matrix, int rows, int cols, int num_type);
float **load_matrix_from_file(int rows, int cols, const char *filename);
float **generate_random_matrix(int rows, int cols, int num_type, float min_val, float max_val);
void clear_input_buffer();
void seperateLine();
void save_multiplication_results(const char *filename, float **matrixA, float **matrixB, float **outputMatrix_single, float **outputMatrix_multi,int m, int n, int p, int num_type, double time_single, double time_multi, int num_iterations);

// clear input buffer after scanf operations
void clear_input_buffer() {
    int characters_in_buffer;
    do {
        characters_in_buffer = getchar();
        
        // exit if reached the end of line or end of file
        if (characters_in_buffer == '\n' || characters_in_buffer == EOF) {
            break;
        }
    } while (1); // repeat until buffer is empty
}

void seperateLine() {
    printf("\n--------------------------------------------\n");
}

// write matrix multiplication results to text file
void save_multiplication_results(const char *outputFilePath, float **matrixA, float **matrixB, float **singleThreadResult, float **multiThreadResult,int rowsA, int columsA, int columsB, int num_type, double singleThreadTime, double multiThreadTime, int num_iterations) {
    
    FILE *file = fopen(outputFilePath, "w");
    if (!file) {
        printf("Error: Cannot create output file %s\n", outputFilePath);
        return;
    }

    fprintf(file, "Matrix Multiplication Results\n");
    fprintf(file, "----------------------------------------\n\n");

    fprintf(file, "Matrix Dimensions:\n");
    fprintf(file, "Matrix A: %d x %d\n", rowsA, columsA);
    fprintf(file, "Matrix B: %d x %d\n", columsA, columsB);
    fprintf(file, "----------------------------------------\n\n");

    // print the performance comparison to the file
    fprintf(file, "Performance Comparison\n");
    fprintf(file, "----------------------------------------\n");
    fprintf(file, "Single-threaded time: %.9f seconds (averaged over %d runs)\n", 
            singleThreadTime, num_iterations);
    fprintf(file, "Multi-threaded time:  %.9f seconds (averaged over %d runs)\n", 
            multiThreadTime, num_iterations);
    
    if (multiThreadTime > 0) {
        fprintf(file, "Speedup Formula = (Single-threaded time / Multi-threaded time)\n");
        fprintf(file, "              = (%.9f / %.9f)\n", singleThreadTime, multiThreadTime);
        fprintf(file, "Speedup: %.3fx\n", singleThreadTime/multiThreadTime);
        fprintf(file, "Performance improvement: %.2f%%\n", ((singleThreadTime/multiThreadTime) - 1) * 100);
    }

    // print that the follwing bellow are the matrices and results
    fprintf(file, "\n----------------------------------------\n\n");
    fprintf(file, "Matrices and Results\n");

    // print the matrix A to the file
    fprintf(file, "Matrix A:\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columsA; j++) {
            if (num_type == TYPE_INTEGER) {
                fprintf(file, "%4d ", (int)matrixA[i][j]);
            } else {
                fprintf(file, "%7.3f ", matrixA[i][j]);
            }
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");

    // print the matrix B to the file
    fprintf(file, "Matrix B:\n");
    for (int i = 0; i < columsA; i++) {
        for (int j = 0; j < columsB; j++) {
            if (num_type == TYPE_INTEGER) {
                fprintf(file, "%4d ", (int)matrixB[i][j]);
            } else {
                fprintf(file, "%7.3f ", matrixB[i][j]);
            }
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");

    // print the results to the file of single thread
    fprintf(file, "Result (Single-threaded):\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columsB; j++) {
            if (num_type == TYPE_INTEGER) {
                fprintf(file, "%4d ", (int)singleThreadResult[i][j]);
            } else {
                fprintf(file, "%7.3f ", singleThreadResult[i][j]);
            }
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");

    // print the results to the file of multi thread
    fprintf(file, "Result (Multi-threaded):\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columsB; j++) {
            if (num_type == TYPE_INTEGER) {
                fprintf(file, "%4d ", (int)multiThreadResult[i][j]);
            } else {
                fprintf(file, "%7.3f ", multiThreadResult[i][j]);
            }
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");

    fclose(file);
    printf("\nResults have been written to %s\n", outputFilePath);
}

// thread function to multiply one row
void *calculate_matrix_row(void *args) {
    // cast the input arguments to thread parameter structure
    ThreadParameters *threadData = (ThreadParameters *)args;
    
    // get the row number of the thread will process
    int currentRow = threadData->row;
    
    // check each column in the result matrix for this row
    for (int resultColumn = 0; resultColumn < threadData->columnsB; resultColumn++) {
        // set the result cell to zero
        threadData->outputMatrix[currentRow][resultColumn] = 0.0;
        
        // calculate dot product for this result cell
        for (int elementIndex = 0; elementIndex < threadData->columnsA; elementIndex++) {
            float firstElement = threadData->matrixA[currentRow][elementIndex];
            float secondElement = threadData->matrixB[elementIndex][resultColumn];
            threadData->outputMatrix[currentRow][resultColumn] += firstElement * secondElement;
        }
    }

    pthread_exit(NULL);
}

// do the matrix multiplication using a single thread
void multiply_single_thread(float **matrixA, float **matrixB, float **outputMatrix, int rowsA, int columsA, int columnsB) {
    // do matrix multiplication
    for (int i = 0; i < rowsA; i++) {
        for (int k = 0; k < columnsB; k++) {
            outputMatrix[i][k] = 0.0;
            for (int j = 0; j < columsA; j++) {
                outputMatrix[i][k] += matrixA[i][j] * matrixB[j][k];
            }
        }
    }
}

// Multi-threaded matrix multiplication
void multiply_multiple_threads(float **matrixA, float **matrixB, float **outputMatrix, int rowsA, int columnsA, int columnsB) {
    // allocate memory for threads
    pthread_t *threads = malloc(rowsA * sizeof(pthread_t));

    // allocate memory for thread parameters
    ThreadParameters *threadParameters = malloc(rowsA * sizeof(ThreadParameters));

    // create threads for each row of the result matrix
    for (int i = 0; i < rowsA; i++) {
        threadParameters[i].matrixA = matrixA;
        threadParameters[i].matrixB = matrixB;
        threadParameters[i].outputMatrix = outputMatrix;
        threadParameters[i].rowsA = rowsA;
        threadParameters[i].columnsA = columnsA;
        threadParameters[i].columnsB = columnsB;
        threadParameters[i].row = i;
        
        if (pthread_create(&threads[i], NULL, calculate_matrix_row, &threadParameters[i]) != 0) {
            printf("Error creating thread %d\n", i);
            exit(1);
        }
    }

    // wait for all threads to finish
    for (int i = 0; i < rowsA; i++) {
        pthread_join(threads[i], NULL);
    }

    // free memory
    free(threads);
    free(threadParameters);
}

// allocates memory for a 2D matrix and initializes it
float **create_matrix(int rows, int columns) {

    // allocate memory for the rows
    float **matrix = malloc(rows * sizeof(float *));

    if (matrix == NULL) {
        printf("Error: Memory allocation failed\n");
        exit(1);
    }
    
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(columns * sizeof(float));
        if (matrix[i] == NULL) {
            printf("Error: Memory allocation failed\n");
            exit(1);
        }
    }
    return matrix;
}

// print the matrix 
void print_matrix(float **matrix, int rows, int columns, int num_type) {

    // print each element of the matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
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
float **generate_random_matrix(int rows, int columns, int num_type, float minimum, float maximum) {
    // create empty matrix with given rows and columns
    float **matrix = create_matrix(rows, columns);

    float range = maximum - minimum;
    
    // fill the matrix with random numbers
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (num_type == TYPE_INTEGER) { // fill the matrix with random integers
                matrix[i][j] = (int)(minimum + (rand() % (int)(range + 1)));
            } 
            else if (num_type == TYPE_FLOAT) { // fill the matrix with random floating point numbers
                matrix[i][j] = minimum + ((float)rand() / RAND_MAX) * range;
            }
            else { // fill the matrix with random integers and floating point numbers
                if (rand() % 2) { 
                    matrix[i][j] = (int)(minimum + (rand() % (int)(range + 1)));
                } else {
                    matrix[i][j] = minimum + ((float)rand() / RAND_MAX) * range;
                }
            }
        }
    }
    return matrix;
}

// load matrix from text file
float **load_matrix_from_file(int rows, int columns, const char *inputFilename) {
    // create empty matrix with given rows and columns
    float **matrix = create_matrix(rows, columns);

    // open the file for reading
    FILE *file = fopen(inputFilename, "r");

    if (!file) { // check if the file is opened successfully
        printf("Error: Cannot open file %s\n", inputFilename);
        exit(1);
    }

    for (int i = 0; i < rows; i++) { // read each element of the matrix and store it in the matrix
        for (int j = 0; j < columns; j++) {
            if (fscanf(file, "%f", &matrix[i][j]) != 1) {
                printf("Error: Reading from file %s failed\n", inputFilename);
                exit(1);
            }
        }
    }

    fclose(file); // close the file

    return matrix;
}

// Free matrix memory
void free_matrix_memory(float **matrix, int rows) {

    // free memory for each row
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(int argc, char *argv[]) {
    seperateLine();
    printf("    Matrix Multiplication Using Threads\n");
    seperateLine();
    
    // check command line for 5 arguments
    if (argc != 5) {
        printf("Give the dimensions of the matrices in this format :\n%s <rows in A> <columns in A> <rows in B> <columns in B>\n", argv[0]);
        printf("Example: %s 5 4 4 2\n", argv[0]);
        return 1;
    }

    // get and validate matrix dimensions
    int rowsA = atoi(argv[1]);
    int columnsA = atoi(argv[2]);
    int rowsB = atoi(argv[3]);
    int columnsB = atoi(argv[4]);

    // check if the dimensions are valid
    if (rowsA <= 0 || columnsA <= 0 || rowsB <= 0 || columnsB <= 0) {
        printf("Error: Matrix dimensions must be positive numbers\n");
        return 1;
    }

    // check if the matrices can be multiplied
    if (columnsA != rowsB) {
        printf("Error: Matrices cannot be multiplied!\n");
        printf("Number of columns in Matrix A (%d) must equal number of rows in Matrix B (%d)\n", 
               columnsA, rowsB);
        return 1;
    }

    // store dimensions for multiplication
    int r_A = rowsA;
    int c_A = columnsA;  
    int c_B = columnsB;

    printf("\nMatrix Dimensions:\n");
    printf("Matrix A: %d x %d\n", rowsA, columnsA);
    printf("Matrix B: %d x %d\n", rowsB, columnsB);
    seperateLine();

    // initialize random seed
    srand(time(NULL));

    float **matrixA = NULL, **matrixB = NULL;
    float **outputMatrix_single = NULL, **outputMatrix_multi = NULL;
    
    // input method selection
    int choice;
    printf("\nChoose input method:\n");
    printf("1. Read matrices from files\n");
    printf("2. Generate random matrices\n");
    printf("Enter choice (1 or 2): ");
    scanf("%d", &choice);
    clear_input_buffer();

    int num_type = TYPE_FLOAT;
    float minimum = 0.0, maximum = 10.0;

    if (choice == 1) {
        char filename[100];
        printf("\nEnter filename for matrix A: ");
        scanf("%s", filename);
        matrixA = load_matrix_from_file(r_A, c_A, filename);
        
        printf("Enter filename for matrix B: ");
        scanf("%s", filename);
        matrixB = load_matrix_from_file(c_A, c_B, filename);
    } 
    else if (choice == 2) {
        seperateLine();
        printf("Choose number type for random generation:\n");
        printf("1. Integers only\n");
        printf("2. Floating point numbers\n");
        printf("3. Mixed (both integers and floating point numbers)\n");
        printf("Enter choice (1-3): ");
        scanf("%d", &num_type);
        clear_input_buffer();

        printf("\nEnter range for random numbers:\n");
        printf("Minimum value: ");
        scanf("%f", &minimum);
        printf("Maximum value: ");
        scanf("%f", &maximum);
        clear_input_buffer();

        printf("\nGenerating random matrices...\n");
        matrixA = generate_random_matrix(r_A, c_A, num_type, minimum, maximum);
        matrixB = generate_random_matrix(c_A, c_B, num_type, minimum, maximum);
    }
    else {
        printf("Invalid choice\n");
        return 1;
    }

    outputMatrix_single = create_matrix(r_A, c_B);
    outputMatrix_multi = create_matrix(r_A, c_B);

    // get number of iterations from user
    int num_iterations;
    printf("\nEnter number of iterations for timing accuracy: ");
    scanf("%d", &num_iterations);
    clear_input_buffer();
    
    if (num_iterations <= 0) {
        printf("Error: Number of iterations must be positive. Using default value of 10.\n");
        num_iterations = 10;
    }

    // do the calculations and measure time with multiple iterations
    struct timeval start, end;
    double time_single = 0.0;
    double time_multi = 0.0;
    
    seperateLine();
    printf("Calculating (Running %d iterations for accurate timing)...\n", num_iterations);

    multiply_single_thread(matrixA, matrixB, outputMatrix_single, r_A, c_A, c_B);
    multiply_multiple_threads(matrixA, matrixB, outputMatrix_multi, r_A, c_A, c_B);
    
    // multiple iterations for single thread
    for (int i = 0; i < num_iterations; i++) {
        gettimeofday(&start, NULL);
        multiply_single_thread(matrixA, matrixB, outputMatrix_single, r_A, c_A, c_B);
        gettimeofday(&end, NULL);
        time_single += (end.tv_sec - start.tv_sec) + 
                      (end.tv_usec - start.tv_usec) / 1000000.0;
    }
    time_single /= num_iterations;


    // multiple iterations for multi thread
    for (int i = 0; i < num_iterations; i++) {
        gettimeofday(&start, NULL);
        multiply_multiple_threads(matrixA, matrixB, outputMatrix_multi, r_A, c_A, c_B);
        gettimeofday(&end, NULL);
        time_multi += (end.tv_sec - start.tv_sec) + 
                     (end.tv_usec - start.tv_usec) / 1000000.0;
    }
    time_multi /= num_iterations;


    // print the results
    seperateLine();
    printf("Results\n");
    seperateLine();
    
    printf("\nMatrix A:\n");
    print_matrix(matrixA, r_A, c_A, num_type);
    
    printf("Matrix B:\n");
    print_matrix(matrixB, c_A, c_B, num_type);
    
    printf("Result (Single-threaded):\n");
    print_matrix(outputMatrix_single, r_A, c_B, num_type);
    
    printf("Result (Multi-threaded):\n");
    print_matrix(outputMatrix_multi, r_A, c_B, num_type);

    // print performance comparison
    seperateLine();
    printf("Performance Comparison\n");
    seperateLine();
    printf("Single-threaded time: %.9f seconds (averaged over %d runs)\n", 
           time_single, num_iterations);
    printf("Multi-threaded time:  %.9f seconds (averaged over %d runs)\n", 
           time_multi, num_iterations);
    
    if (time_multi > 0) {
        printf("Speedup Formula = (Single-threaded time / Multi-threaded time)\n");
        printf("              = (%.9f / %.9f)\n", time_single, time_multi);
        printf("Speedup: %.3fx\n", time_single/time_multi);
        printf("Performance improvement: %.2f%%\n", ((time_single/time_multi) - 1) * 100);
    }

    // if user want to save the results to a file
    printf("\nDo you want to save the results to a file? (1: Yes, 0: No): ");
    int save_choice;
    scanf("%d", &save_choice);
    clear_input_buffer();

    if (save_choice == 1) {
        char output_filename[100];
        printf("Enter output filename (without .txt extension): ");
        scanf("%s", output_filename);
        
        // add .txt extension to the end of the filename 
        char full_filename[104];
        snprintf(full_filename, sizeof(full_filename), "%s.txt", output_filename);
        
        save_multiplication_results(full_filename, matrixA, matrixB, outputMatrix_single, outputMatrix_multi,r_A, c_A, c_B, num_type, time_single, time_multi, num_iterations);
    }

    // after the calculations free the memory
    free_matrix_memory(matrixA, r_A);
    free_matrix_memory(matrixB, c_A);
    free_matrix_memory(outputMatrix_single, r_A);
    free_matrix_memory(outputMatrix_multi, r_A);

    return 0;
}