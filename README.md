# Matrix Multiplication Using POSIX Threads

This project implements a matrix multiplication system using POSIX threads in C. The program offers both single-threaded and multi-threaded approaches to matrix multiplication, allowing users to compare performance differences between sequential and parallel implementations. At its core, the program divides the multiplication task among multiple threads, with each thread responsible for computing one row of the resultant matrix.

## Implementation Overview

The matrix multiplication program is designed around a thread-per-row approach, where each thread is responsible for computing one complete row of the result matrix. This design choice was made to minimize thread synchronization overhead while still achieving parallel computation benefits. The program uses dynamically allocated 2D arrays to store matrices, allowing for flexible matrix sizes limited only by available system memory.

## Repository Setup and Usage

### First Time Setup
**Clone the Repository**
   ```bash
   git clone https://github.com/melkor/matrix-multiplication.git
   cd matrix-multiplication
   ```

### Updating Your Local Copy
  ```bash
  git pull origin main
  ```

### Matrix Multiplication Implementation

The program implements two distinct multiplication methods:

1. **Single-Threaded Multiplication**
  ```c
  void multiply_single_thread(float **matrixA, float **matrixB, float **outputMatrix, 
                            int rowsA, int columsA, int columnsB) {
      for (int i = 0; i < rowsA; i++) {
          for (int k = 0; k < columnsB; k++) {
              outputMatrix[i][k] = 0.0;
              for (int j = 0; j < columsA; j++) {
                  outputMatrix[i][k] += matrixA[i][j] * matrixB[j][k];
              }
          }
      }
  }
  ```


2. **Multi-Threaded Multiplication**
The parallel implementation creates one thread per row of the result matrix:

  ```c
  void multiply_multiple_threads(float **matrixA, float **matrixB, float **outputMatrix, 
                               int rowsA, int columnsA, int columnsB) {
      pthread_t *threads = malloc(rowsA * sizeof(pthread_t));
      ThreadParameters *threadParameters = malloc(rowsA * sizeof(ThreadParameters));
  
      for (int i = 0; i < rowsA; i++) {
          // Initialize thread parameters
          threadParameters[i].matrixA = matrixA;
          threadParameters[i].matrixB = matrixB;
          threadParameters[i].outputMatrix = outputMatrix;
          threadParameters[i].row = i;
          // ... other parameter initialization ...
          
          pthread_create(&threads[i], NULL, calculate_matrix_row, &threadParameters[i]);
      }
  
      // Wait for all threads to complete
      for (int i = 0; i < rowsA; i++) {
          pthread_join(threads[i], NULL);
      }
  }
  ```

## Testing the Program

The program offers three ways to test matrix multiplication: using sample files, creating custom test cases, or generating random matrices. Below are detailed instructions for each method.

### 1. Using Sample Files

The repository includes sample matrix files for testing. Here's how to use them:

  ```bash
  # Run the program with dimensions matching the sample files
  ./matrix_multiplication 5 6 6 6
  
  # When prompted for input method, choose option 1 (Read from files)
  Enter choice (1 or 2): 1
  
  # Enter the sample filenames
  Enter filename for matrix A: matrixA.txt
  Enter filename for matrix B: matrixB.txt
  ```

Sample file contents (matrixA.txt):
  ```
  12.5 7.0 15.3 22.0 9.1 4.0
  8.0 16.7 3.0 11.2 19.0 5.8
  21.0 6.4 13.0 2.9 17.0 10.0
  4.7 18.0 9.6 14.2 1.0 23.0
  15.0 7.8 20.0 5.3 12.6 8.0
  ```

### 2. Creating Custom Test Cases

You can create your own test matrices following these steps:

1. Create text files for your matrices:
   The file input system allows users to load matrices from text files. The implementation handles both integer and floating-point numbers automatically:
   
    ```c
    float **load_matrix_from_file(int rows, int cols, const char *filename) {
        float **matrix = create_matrix(rows, cols);
        FILE *file = fopen(filename, "r");
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fscanf(file, "%f", &matrix[i][j]);
            }
        }
        return matrix;
    }
    ```
   - Use space-separated values
   - Each row on a new line
   - Can use both integers and decimal numbers
   - Ensure consistent dimensions

Example custom matrices:
  ```bash
  # customA.txt (2x3 matrix)
  1 2 3
  4 5 6
  
  # customB.txt (3x2 matrix)
  7 8
  9 10
  11 12
  
  # Run with matching dimensions
  ./matrix_multiplication 2 3 3 2
  ```

### 2. Using Random Matrix Generation

For quick testing with random values:

   ```bash
   # Example: Create 4x3 and 3x5 matrices
   ./matrix_multiplication 4 3 3 5
   
   # Choose random generation
   Enter choice (1 or 2): 2
   
   # Available number types:
   # 1. Integers only (e.g., 1, 2, 3)
   # 2. Floating point (e.g., 1.5, 2.7, 3.9)
   # 3. Mixed (combination of both)
   Enter choice (1-3): 1
   
   # Set your desired range
   Minimum value: 1
   Maximum value: 100
   ```

The program will:
1. Generate random matrices within your specifications
2. Perform multiplication using both single and multi-threaded methods
3. Display the matrices and results
4. Show performance comparison

## Usage Instructions

### Linux
1. **Installation of Prerequisites**
   ```bash
   sudo apt-get update
   sudo apt-get install gcc make
   ```

2. **Compilation**
   ```bash
   gcc -o matrix_multiplication matrix_multiplication.c -pthread -lm
   ```

3. **Execution**
   ```bash
   ./matrix_multiplication <rows_A> <cols_A> <rows_B> <cols_B>
   ```

### macOS
1. **Installation of Prerequisites**
   - Install Xcode Command Line Tools:
     ```bash
     xcode-select --install
     ```
   - Or install full Xcode from the App Store

2. **Compilation**
   ```bash
   gcc -o matrix_multiplication matrix_multiplication.c -pthread
   ```

3. **Execution**
   ```bash
   ./matrix_multiplication <rows_A> <cols_A> <rows_B> <cols_B>
   ```

### Windows
1. **Installation of Prerequisites**
   - Method 1: Using MinGW
     1. Download and install MinGW from [MinGW Website](https://www.mingw-w64.org/)
     2. Add MinGW's bin directory to system PATH
   
   - Method 2: Using WSL (Windows Subsystem for Linux)
     1. Enable WSL from Windows Features
     2. Install Ubuntu or preferred Linux distribution from Microsoft Store
     3. Follow Linux instructions within WSL

2. **Compilation**
   - Using MinGW:
     ```cmd
     gcc -o matrix_multiplication.exe matrix_multiplication.c -pthread
     ```
   - Using WSL:
     ```bash
     gcc -o matrix_multiplication matrix_multiplication.c -pthread -lm
     ```

3. **Execution**
   - Using MinGW:
     ```cmd
     matrix_multiplication.exe <rows_A> <cols_A> <rows_B> <cols_B>
     ```
   - Using WSL:
     ```bash
     ./matrix_multiplication <rows_A> <cols_A> <rows_B> <cols_B>
     ```

## Performance Considerations

The program's performance characteristics vary based on matrix size and system capabilities:

- For small matrices (< 10x10), the multi-threaded version might be slower due to thread creation overhead
- Thread performance generally improves with larger matrices due to better parallelization benefits
- The program automatically handles memory management to prevent leaks

## Error Handling and Validation

The program implements comprehensive error checking:

   ```c
   if (columnsA != rowsB) {
       printf("Error: Matrices cannot be multiplied!\n");
       printf("Number of columns in Matrix A (%d) must equal number of rows in Matrix B (%d)\n", 
              columnsA, rowsB);
       return 1;
   }
   ```

The system validates:
- Matrix dimensions and multiplicability
- Memory allocation success
- File operations
- Thread creation and management

Each operation includes appropriate error messages and graceful error handling to prevent crashes and memory leaks.


## Performance Measurement and Output

The program implements sophisticated timing mechanisms to accurately measure performance differences:

   ```c
   struct timeval start, end;
   double time_single = 0.0, time_multi = 0.0;
   
   // Multiple iterations for accurate timing
   for (int i = 0; i < num_iterations; i++) {
       gettimeofday(&start, NULL);
       multiply_single_thread(/*...*/);
       gettimeofday(&end, NULL);
       time_single += (end.tv_sec - start.tv_sec) + 
                     (end.tv_usec - start.tv_usec) / 1000000.0;
   }
   ```

Results can be saved to a file, including:
- Complete input matrices
- Result matrices from both methods
- Detailed timing information
- Performance comparisons and speedup calculations


### Common Issues and Solutions
1. **Windows pthread Error**: If using MinGW and encountering pthread-related errors:
   - Install the pthreads-win32 package
   - Add `-lpthread` instead of `-pthread` in compilation

2. **Permission Denied**: On Unix-based systems (Linux/macOS):
   ```bash
   chmod +x matrix_multiplication
   ```

3. **Math Library Error**: If encountering math library errors:
   - Linux/WSL: Add `-lm` flag during compilation
   - macOS: No additional flag needed
   - Windows: Add `-lm` when using MinGW


### Contributing
1. **Fork the Repository**
   - Click the 'Fork' button on GitHub
   - Clone your forked repository

2. **Make Changes**
   - Create a new branch for your feature
     ```bash
     git checkout -b feature-name
     ```
   - Make your changes
   - Test thoroughly
   - Commit with clear messages
     ```bash
     git add .
     git commit -m "Description of changes"
     ```

3. **Submit Changes**
   - Push to your fork
     ```bash
     git push origin feature-name
     ```
   - Create a Pull Request on GitHub
   - Provide clear description of changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.
