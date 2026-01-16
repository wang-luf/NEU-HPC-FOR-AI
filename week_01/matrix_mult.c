#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    int M; 
    int K; 
    int N; 
    double *A; 
    double *B; 
    double *C;
} ThreadArgs;


void* thread_worker(void* arg) {
    ThreadArgs* data = (ThreadArgs*) arg;

    int M = data->M;
    int K = data->K;
    int N = data->N;
    int start = data->start_row;
    int end = data->end_row;

    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int p = 0; p < K; p++) {

                sum += data->A[i * K + p] * data->B[p * N + j];
            }
            data->C[i * N + j] = sum;
        }
    }

    pthread_exit(NULL);
}

void parallel_multiply(double* A, double* B, double* C, int M, int K, int N, int num_threads) {
    pthread_t threads[num_threads];
    ThreadArgs thread_args[num_threads];

    if (num_threads > M) {
        num_threads = M;
    }

    int rows_per_thread = M / num_threads;

    for (int t = 0; t < num_threads; t++) {
        thread_args[t].thread_id = t;
        thread_args[t].M = M;
        thread_args[t].K = K;
        thread_args[t].N = N;
        thread_args[t].A = A;
        thread_args[t].B = B;
        thread_args[t].C = C;
        
        thread_args[t].start_row = t * rows_per_thread;
        
        if (t == num_threads - 1) {
            thread_args[t].end_row = M;
        } else {
            thread_args[t].end_row = (t + 1) * rows_per_thread;
        }

        int rc = pthread_create(&threads[t], NULL, thread_worker, (void *)&thread_args[t]);
        if (rc) {
            printf("Error: unable to create thread, %d\n", rc);
            exit(-1);
        }
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}


void init_matrix(double* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (double)(rand() % 10); 
    }
}

int main() {

    printf("=== Starting Functional Tests (Corner Cases) ===\n");
    
    int test_cases[][3] = {
        {1, 1, 1},
        {1, 1, 5},
        {2, 1, 3},
        {2, 2, 2},
        {10, 20, 10} 
    };
    
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);

    for (int t = 0; t < num_cases; t++) {
        int M = test_cases[t][0];
        int K = test_cases[t][1];
        int N = test_cases[t][2];

        double *A = (double*)malloc(M * K * sizeof(double));
        double *B = (double*)malloc(K * N * sizeof(double));
        double *C = (double*)malloc(M * N * sizeof(double));
        
        init_matrix(A, M, K);
        init_matrix(B, K, N);

        parallel_multiply(A, B, C, M, K, N, 2);
        
        printf("[PASS] Test Case %d: %dx%d * %dx%d = %dx%d matrix calculated.\n", 
               t+1, M, K, K, N, M, N);

        free(A); free(B); free(C);
    }
    printf("=== All Functional Tests Passed ===\n\n");


    printf("=== Starting Performance Benchmark ===\n");
    int M = 1024, K = 1024, N = 1024;
    printf("Matrix Size: %dx%d multiplied by %dx%d\n", M, K, K, N);
    
    double *A = (double*)malloc(M * K * sizeof(double));
    double *B = (double*)malloc(K * N * sizeof(double));
    double *C = (double*)malloc(M * N * sizeof(double));
    
    init_matrix(A, M, K);
    init_matrix(B, K, N);

    int thread_counts[] = {1, 4, 16, 32, 64, 128};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_configs; i++) {
        int th_count = thread_counts[i];
        
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        parallel_multiply(A, B, C, M, K, N, th_count);

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double time_taken = (end.tv_sec - start.tv_sec) + 
                            (end.tv_nsec - start.tv_nsec) * 1e-9;
        
        printf("Threads: %3d | Time: %.4f seconds\n", th_count, time_taken);
    }

    free(A); free(B); free(C);

    return 0;
}