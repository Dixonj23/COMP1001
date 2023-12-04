/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <omp.h>

#define M 1024*512
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N 8192
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1


//function declaration
void initialize();
void routine1(float alpha, float beta);
void routine2(float alpha, float beta);

void routine1_vec(float alpha, float beta, float* y, const float* z);
void routine2_vec(float alpha, float beta);

__declspec(align(64)) float  y[M], z[M] ;
__declspec(align(64)) float A[N][N], x[N], w[N];

int main() {

    float alpha = 0.023f, beta = 0.045f;
    double run_time, start_time;
    unsigned int t;

    initialize();

    printf("\nRoutine1:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));

    
    /*
        CHECKING VECTORISATION
    */
    
    
    printf("\nRoutine1_vec:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1_vec(alpha, beta, y, z);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2_vec:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2_vec(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));



    return 0;
}

void initialize() {

    unsigned int i, j;

    //initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013f;
        }

    //initialize routine1 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
    }


}




void routine1(float alpha, float beta) {

    unsigned int i;


    for (i = 0; i < M; i++)
        y[i] = alpha * y[i] + beta * z[i];

}


void routine2(float alpha, float beta) {

    unsigned int i, j;


    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            w[i] = w[i] - beta + alpha * A[i][j] * x[j];


}


void routine1_vec(float alpha, float beta, float* y, const float* z) {
    const int simdSize = 8;

    for (int i = 0; i < M; i += 8) {
        __m256 y_vec = _mm256_load_ps(&y[i]);
        __m256 z_vec = _mm256_load_ps(&z[i]);

        // Multiply alpha and y, multiply beta and z
        __m256 alpha_y = _mm256_mul_ps(_mm256_set1_ps(alpha), y_vec);
        __m256 beta_z = _mm256_mul_ps(_mm256_set1_ps(beta), z_vec);

        // Add the results
        __m256 result = _mm256_add_ps(alpha_y, beta_z);

        // Store the result back to the y array
        _mm256_store_ps(&y[i], result);

    }



}

void routine2_vec(float alpha, float beta) {
    const int simdSize = 8;

    for (int i = 0; i < N; ++i) {
        __m256 w_vec = _mm256_set1_ps(w[i]);
        for (int j = 0; j < N; j += simdSize) {
            __m256 A_vec = _mm256_loadu_ps(&A[i][j]); // Load 8 elements from A[i][j]

            __m256 x_vec = _mm256_loadu_ps(&x[j]); // Load 8 elements from x

            __m256 mul_result = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(alpha), A_vec), x_vec); // alpha * A[i][j] * x[j]

            w_vec = _mm256_sub_ps(w_vec, _mm256_sub_ps(_mm256_set1_ps(beta), mul_result)); // w[i] - beta + alpha * A[i][j] * x[j]
        }
        float result[8];
        _mm256_storeu_ps(result, w_vec); // Store the result back to the temporary array

        for (int k = 0; k < simdSize; ++k) {
            w[i] = result[k]; // Store back to w array
        }
    }
    
}