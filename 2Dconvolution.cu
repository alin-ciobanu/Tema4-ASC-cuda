/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>
#include <helper_timer.h>
#include <helper_functions.h>
#include <helper_math.h>

// includes, project
#include "2Dconvolution.h"


////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(int width, int height);
Matrix AllocateMatrix(int width, int height);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
void ConvolutionOnDeviceShared(const Matrix M, const Matrix N, Matrix P);

void printMatrix(Matrix m) {
	int i, j;
	for (i = 0; i < m.height; i++) {
		for (j = 0; j < m.width; j++) {
			printf("%f ", m.elements[i * m.height + m.width]);
		}
		printf("\n");
	}
	printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
// Înmulțirea fără memorie partajată
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;
	int m, n;

	if (row >= 0 && row < N.height && col >= 0 && col < N.width) {

		for (m = 0; m < KERNEL_SIZE; m++) {
			for (n = 0; n < KERNEL_SIZE; n++) {
				float Nelem;
				if ((row + m - 2 < 0) || (row + m - 2 >= N.height) || 
					(col + n - 2 < 0) || (col + n - 2 >= N.width)) {
					Nelem = 0.0;
				}
				else {
					Nelem = N.elements[(row + m - 2) * N.width + (col + n - 2)];
				}
				Pvalue += M.elements[m * M.width + n] * Nelem;
			}
		}
		P.elements[row * P.width + col] = Pvalue;
	}
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.pitch = A.pitch;
	Asub.elements = &A.elements[A.pitch * (BLOCK_SIZE + 2) * row + (BLOCK_SIZE + 2) * col];

	return Asub;
}

////////////////////////////////////////////////////////////////////////////////
// Înmulțirea cu memorie partajată
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernelShared(Matrix M, Matrix N, Matrix P)
{

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	
	Matrix Psub = GetSubMatrix(P, blockRow, blockCol);
	Matrix Nsub = GetSubMatrix(P, blockRow, blockCol);
	Matrix Msub = GetSubMatrix(M, blockRow, blockCol);

	float Pvalue = 0.0;
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int m, n;
	
	if (row >= 0 && row < N.height && col >= 0 && col < N.width) {

		for (m = 0; m < KERNEL_SIZE; m++) {
			for (n = 0; n < KERNEL_SIZE; n++) {
				__shared__ float Nelem;
				if ((row + m - 2 < 0) || (row + m - 2 >= N.height) || 
					(col + n - 2 < 0) || (col + n - 2 >= N.width)) {
					Nelem = 0.0;
				}
				else {
					Nelem = N.elements[(row + m - 2) * N.width + (col + n - 2)];
				}
				Pvalue += M.elements[m * M.width + n] * Nelem;
			}
		}
		P.elements[row * P.width + col] = Pvalue;
	}

}

////////////////////////////////////////////////////////////////////////////////
// Returnează 1 dacă matricele sunt ~ egale
////////////////////////////////////////////////////////////////////////////////
int CompareMatrices(Matrix A, Matrix B)
{
    int i;
    if(A.width != B.width || A.height != B.height || A.pitch != B.pitch)
        return 0;
    int size = A.width * A.height;
    for(i = 0; i < size; i++)
        if(fabs(A.elements[i] - B.elements[i]) > MAX_ERR)
            return 0;
    return 1;
}
void GenerateRandomMatrix(Matrix m)
{
    int i;
    int size = m.width * m.height;

    srand(time(NULL));

    for(i = 0; i < size; i++)
        m.elements[i] = rand() / (float)RAND_MAX;
}

////////////////////////////////////////////////////////////////////////////////
// main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    int width = 0, height = 0;
    FILE *f, *out;
    if(argc < 2)
    {
        printf("Argumente prea puține, trimiteți id-ul testului care trebuie rulat\n");
        return 0;
    }
    char name[100];
    sprintf(name, "./tests/test_%s.txt", argv[1]);
    f = fopen(name, "r");
    out = fopen("out.txt", "a");
    fscanf(f, "%d%d", &width, &height);
    Matrix M;//kernel de pe host
    Matrix N;//matrice inițială de pe host
    Matrix P;//rezultat fără memorie partajată calculat pe GPU
    Matrix PS;//rezultatul cu memorie partajată calculat pe GPU
    
    M = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE);
    N = AllocateMatrix(width, height);        
    P = AllocateMatrix(width, height);
    PS = AllocateMatrix(width, height);

    GenerateRandomMatrix(M);
    GenerateRandomMatrix(N);
    

    // M * N pe device
    ConvolutionOnDevice(M, N, P);
    
    // M * N pe device cu memorie partajată
    ConvolutionOnDeviceShared(M, N, PS);

    // calculează rezultatul pe CPU pentru comparație
    Matrix reference = AllocateMatrix(P.width, P.height);
    computeGold(reference.elements, M.elements, N.elements, N.height, N.width);
        
    // verifică dacă rezultatul obținut pe device este cel așteptat
    int res = CompareMatrices(reference, P);
	
    printf("Test global %s\n", (1 == res) ? "PASSED" : "FAILED");
    fprintf(out, "Test global %s %s\n", argv[1], (1 == res) ? "PASSED" : "FAILED");

    // verifică dacă rezultatul obținut pe device cu memorie partajată este cel așteptat
  //  int ress = CompareMatrices(reference, PS);
    int ress = CompareMatrices(reference, PS);

    printf("Test shared %s\n", (1 == ress) ? "PASSED" : "FAILED");
    fprintf(out, "Test shared %s %s\n", argv[1], (1 == ress) ? "PASSED" : "FAILED");
   
    // Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
    FreeMatrix(&PS);

    fclose(f);
    fclose(out);
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    Matrix Md, Nd, Pd; //matricele corespunzătoare de pe device

    //pentru măsurarea timpului de execuție în kernel
    StopWatchInterface *kernelTime = NULL;
    sdkCreateTimer(&kernelTime);
    sdkResetTimer(&kernelTime);

	Md = AllocateDeviceMatrix(M.width, M.height);
	Nd = AllocateDeviceMatrix(N.width, N.height);
	Pd = AllocateDeviceMatrix(P.width, P.height);

	cudaMemcpy(Md.elements, M.elements, M.width * M.height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Nd.elements, N.elements, N.width * N.height * sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(N.width / dimBlock.x + 1, N.height / dimBlock.y + 1);

    sdkStartTimer(&kernelTime);
	ConvolutionKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

    cudaThreadSynchronize();
    sdkStopTimer(&kernelTime);
    printf ("Timp execuție kernel: %f ms\n", sdkGetTimerValue(&kernelTime));

	cudaMemcpy(P.elements, Pd.elements, Pd.width * Pd.height * sizeof(float), cudaMemcpyDeviceToHost);
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}


void ConvolutionOnDeviceShared(const Matrix M, const Matrix N, Matrix P)
{
    Matrix Md, Nd, Pd; //matricele corespunzătoare de pe device

    //pentru măsurarea timpului de execuție în kernel
    StopWatchInterface *kernelTime = NULL;
    sdkCreateTimer(&kernelTime);
    sdkResetTimer(&kernelTime);

	Md = AllocateDeviceMatrix(M.width, M.height);
	Nd = AllocateDeviceMatrix(N.width, N.height);
	Pd = AllocateDeviceMatrix(P.width, P.height);

	cudaMemcpy(Md.elements, M.elements, M.width * M.height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Nd.elements, N.elements, N.width * N.height * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(N.width / dimBlock.x + 1, N.height / dimBlock.y + 1);

    sdkStartTimer(&kernelTime);
	ConvolutionKernelShared<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
    cudaThreadSynchronize();
    sdkStopTimer(&kernelTime);
    printf ("Timp execuție kernel cu memorie partajată: %f ms\n", sdkGetTimerValue(&kernelTime));
    cudaMemcpy(P.elements, Pd.elements, Pd.width * Pd.height * sizeof(float), cudaMemcpyDeviceToHost);
	FreeDeviceMatrix(&Md);
	FreeDeviceMatrix(&Nd);
	FreeDeviceMatrix(&Pd);
}


// Alocă o matrice de dimensiune height*width pe device
Matrix AllocateDeviceMatrix(int width, int height)
{
    Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = width * height;
	cudaMalloc( (void**) &(M.elements), size * sizeof(float));
    return M;
}

// Alocă matrice pe host de dimensiune height*width
Matrix AllocateMatrix(int width, int height)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;    
    M.elements = (float*) malloc(size*sizeof(float));
    return M;
}    

// Eliberează o matrice de pe device
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Eliberează o matrice de pe host
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}