#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cstdlib>
#include <sstream>
#include <iomanip>

using namespace std;

//input, filter and output considering maximum dimension possible
double a[1000][1000], h[10][10], c[1000][1000];

/**
 * CUDA Kernel Device code
 *
 * Computes the 2D convolution of input b with filter h and output c
 */

 __global__ void
 convolution2D(double *a_1, double *h_1, double *c_1, int rows_a, int columns_a, int rows_h, int columns_h)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    double sum;

    if((idx < (columns_h+columns_a - 1)) && (idy < (rows_h + rows_a - 1)))
    {
        sum = 0;
        for(int k=0; k< rows_h; k++)
        {
            for(int j=0; j< columns_h; j++)
            {
                if ((idy - k) >= 0 && (idx - j) >= 0 && (idy - k) < rows_a && (idx - j) < columns_a) 
                {
                    sum += a_1[((idy - k)*columns_a) + (idx - j)] * h_1[k*columns_h + j];
                }

            }
        }
        c_1[idy*(columns_a + columns_h - 1) + idx] = sum;
        __syncthreads();
    }
}

/**
 * Host main routine
 */
 int main(int argc, char** argv)
 {
     // Error code to check return values for CUDA calls
     cudaError_t err = cudaSuccess;

     int columns_a, columns_h, rows_a, rows_h, size_a, size_h;
     columns_a = columns_h = 0;
     size_a = size_h = rows_a = rows_h = 0;
     char *ip_file;
     string line;
     double input;
     ip_file = argv[1];
     int i, j, k;

     ifstream file(ip_file);
     if(file.is_open())
     {
         i=0;
         while(getline(file, line) && line != "")
         {
              j=0;
             stringstream ss(line);
             while(ss >> input)
             {
                 a[i][j] = input;
                 size_a++;
                 j++;
             }
             i++;
             rows_a++;
         }
         
         k=0;
         while(getline(file, line) && line != "")
         {
              j=0;
             stringstream ss(line);
             while(ss >> input)
             {
                 h[k][j] = input;
                 j++;
                 size_h++;
             }
             k++;
             rows_h++;
         }
     }
     file.close();
     columns_a = size_a/rows_a;
     columns_h = size_h/rows_h;
     int op_size = ((rows_a+rows_h-1)*(columns_a+columns_h-1));
     size_t size_ax = size_a*sizeof(double);
     size_t size_hx = size_h*sizeof(double);
     size_t size_cx = op_size*sizeof(double);
     
     // Allocate the host input vector a
     double *h_a = (double *)malloc(size_ax);
 
     // Allocate the host input vector h
     double *h_h = (double *)malloc(size_hx);
 
     // Allocate the host output vector c
     double *h_c = (double *)malloc(size_cx);

     // Verify that allocations succeeded
     if (h_a == NULL || h_h == NULL || h_c == NULL)
     {
         fprintf(stderr, "Failed to allocate host vectors!\n");
         exit(EXIT_FAILURE);
     }
     for (int i = 0; i < rows_a; i++)
     {
         for(int j=0;j< columns_a; j++)
         {
            h_a[i*columns_a + j] = a[i][j];
         }
     }

     for (int i = 0; i < rows_h; i++)
     {
         for(int j=0;j< columns_h; j++)
         {
            h_h[i*columns_h + j] = h[i][j];
         }
     }

     for (int i = 0; i < op_size; i++)
    {
        h_c[i] = rand()/(double)RAND_MAX;
    }
     //Allocate the device inputs
     double *d_a = NULL, *d_h = NULL, *d_c = NULL;
     err = cudaMalloc((void **)&d_a, size_ax);
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector a (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
     err = cudaMalloc((void **)&d_h, size_hx);
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector h (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     err = cudaMalloc((void **)&d_c, size_cx);
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector c (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
     // Copy the host input vectors a and h in host memory to the device input vectors in
     // device memory
     //printf("Copy input data from the host memory to the CUDA device\n");
     err=cudaMemcpy(d_a, h_a, size_ax, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector a from host to device (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     err = cudaMemcpy(d_h, h_h, size_hx, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
     // Launch the CUDA Kernel
     dim3 blocksPerGrid(((rows_a + rows_h - 2) / 32) + 1, ((columns_h + columns_a - 2) / 32) + 1, 1);
     dim3 threadsPerBlock(32,32,1);
     convolution2D<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_h, d_c, rows_a, columns_a, rows_h, columns_h);
     err = cudaGetLastError();
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to launch convolution2D kernel (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     err = cudaDeviceSynchronize();
     if (err != cudaSuccess)
	 {
		fprintf(stderr, "Failed to synchronize (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	 }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
     // Copy the device result in device memory to the host result in host memory.
     err = cudaMemcpy(h_c, d_c, size_cx, cudaMemcpyDeviceToHost);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector c from device to host (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     for (int i = 0; i < rows_a + rows_h - 1; i++) {
        for (int j = 0; j < columns_a + columns_h - 1; j++) 
        {	
			cout << fixed << setprecision(3) << h_c[(i*(columns_h+columns_a-1))+j] << " ";
		}
		cout << endl;
	}
 
     // Free device global memory
     err = cudaFree(d_a);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device vector a (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     err = cudaFree(d_h);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device vector h (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     err = cudaFree(d_c);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device vector c (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Free host memory
     free(h_a);
     free(h_h);
     free(h_c);
 
     // Reset the device and exit
     // cudaDeviceReset causes the driver to clean up all state. While
     // not mandatory in normal operation, it is good practice.  It is also
     // needed to ensure correct operation when the application is being
     // profiled. Calling cudaDeviceReset causes all profile data to be
     // flushed before the application exits
     err = cudaDeviceReset();
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     printf("Done\n");
     return 0;
 }
