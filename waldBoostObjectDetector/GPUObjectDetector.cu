
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "npp.h"
#include<curand.h>
#include<curand_kernel.h>

#include "WBSSettings.h"

#include <stdio.h>

cudaError_t wbsOnCuda(double*, int*, int, int, double*, int, int *, int);

__global__ void applyClassifiersKernel(double *imageData,  int *imageDataDescriptors, double *weakClassProv, int descriptorStep, int imageSize, int* numberOfThreadsInImage, int weakClassNum)
{
    int threadNumber = threadIdx.x;
	int blockNumber = blockIdx.x;
	int gridNumber = gridDim.x;

	int realThreadNumber = threadNumber + blockNumber*WARP_SIZE;

	int imageNumber = 0;
	int threadNumberOnThisImage = 0;

	// find thread number on this image
	int previousNOT = 0;
	for (int i = 0; i < descriptorStep; i++) {		
		int currentNOT = numberOfThreadsInImage[i];
		if (realThreadNumber >= previousNOT && realThreadNumber < currentNOT) {
			// we are in right image
			imageNumber = i;
			threadNumberOnThisImage = realThreadNumber - previousNOT;
			break;
		}
		previousNOT = currentNOT;
	}
	
	// find out which data we will need
	/*int width;
	int height;
	int dataDescriptorNumber;
	int threadNumberOnThisImage;
	for (int i = 0; i < descriptorStep; i++) {
		int tN = 0;
		int nexttN = 0;
		int w = imageDataDescriptors[i + descriptorStep];
		int h = 0;
		if (i + 1 >= descriptorStep) {
			height = (imageSize - imageDataDescriptors[i]) / width;
		}
		else {
			height = (imageDataDescriptors[i + 1] - imageDataDescriptors[i]) / width;
		}
		nexttN += ((width - SLIDING_WINDOW_SIZE) / SLIDING_WINDOW_STEP) * ((height - SLIDING_WINDOW_SIZE) / SLIDING_WINDOW_STEP);
		// if number of threads which need to work is bigger than the thread number among all threads, we start calculating with this data
		if (nexttN > realThreadNumber) {
			dataDescriptorNumber = i;
			width = w;
			height = h;
			threadNumberOnThisImage = realThreadNumber - tN;
			break;
		}
		tN = nexttN;
	}*/

	// get descriptors of data
	int dataStart = imageDataDescriptors[imageNumber];
	int width = imageDataDescriptors[imageNumber + descriptorStep];
	int realWidthDiff = imageDataDescriptors[imageNumber + 2*descriptorStep];
	int dataEnd = imageDataDescriptors[imageNumber + 3 * descriptorStep];

	// find out on which position the sliding window should be
	int windowNumber = threadNumberOnThisImage * SLIDING_WINDOW_STEP;
	int windowsOnLine = (width - SLIDING_WINDOW_SIZE) - SLIDING_WINDOW_STEP;

	windowNumber = windowNumber * SLIDING_WINDOW_STEP;

	int lineNumber = windowNumber / windowsOnLine;
	int columnNumber = windowNumber % windowsOnLine;

	// drop the area with some chance
	if (weakClassProv[threadNumberOnThisImage%weakClassNum] < 0.99) {
		return;
	}

	//imageData[dataStart+columnNumber + width*lineNumber] = 255.0;

	// this part was able to go through classifier, draw rectangel - set boundary to 255
	int rowNumber = lineNumber;
	int colNumber = columnNumber;
	for (int j = 0; j < SLIDING_WINDOW_SIZE; j++) {
		// draw horizontal lines
		imageData[dataStart + colNumber + width*rowNumber+j] = 255.0;
		imageData[dataStart + colNumber + width*rowNumber + j + SLIDING_WINDOW_SIZE*width] = 255.0;

		// draw vertical lines
		imageData[dataStart + colNumber + width*rowNumber + j*width] = 255.0;
		imageData[dataStart + colNumber + width*rowNumber + j*width+SLIDING_WINDOW_SIZE] = 255.0;
	}
}

int GPUObjectDetector(double* imageData, int* imageDataDescriptor, int imageDataDescriptorStep, int imageDataSize, double* weakClassProb, int weakClassNum)
{
	// create array which will deternime how much threads should work on which image from pyramid
	int *threadsOnImage = new int[imageDataDescriptorStep];
	int finalNumberOfThreads = 0;
	for (int i = 0; i < imageDataDescriptorStep; i++) {
		//fprintf(stdout, "Desciptor value: %d, %d, %d, %d\n", descriptors[i], descriptors[i + descriptorStep], descriptors[i + 2 * descriptorStep], descriptors[i + 3 * descriptorStep]);
		int width = imageDataDescriptor[i + imageDataDescriptorStep];
		int height = 0;
		if (i + 1 >= imageDataDescriptorStep) {
			height = (imageDataSize - imageDataDescriptor[i]) / width;
		}
		else {
			height = (imageDataDescriptor[i + 1] - imageDataDescriptor[i]) / width;
		}
		int threadsOnThisImage = ((width - SLIDING_WINDOW_SIZE) / SLIDING_WINDOW_STEP) * ((height - SLIDING_WINDOW_SIZE) / SLIDING_WINDOW_STEP);
		finalNumberOfThreads += threadsOnThisImage;
		threadsOnImage[i] = finalNumberOfThreads;
	}

    cudaError_t cudaStatus = wbsOnCuda(imageData, imageDataDescriptor, imageDataDescriptorStep, imageDataSize, weakClassProb, weakClassNum, threadsOnImage, finalNumberOfThreads);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to detect object using waldboost.
cudaError_t wbsOnCuda(double *image, int *descriptors, int descriptorStep, int imageSize, double* weakClassProb, int weakClassnum, int *threadsOnImage, int numberOfThread)
{
	double *dev_image = 0;
	int *dev_descriptors = 0;
	double *dev_weakClassProb = 0;
	int *dev_numOfThreadInImage = 0;
    cudaError_t cudaStatus;

	// number of cuda devices
	int cudaDeviceCount;
	cudaStatus = cudaGetDeviceCount(&cudaDeviceCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	fprintf(stdout, "number of CUDA devices: %d\n", cudaDeviceCount);
	for (int i = 0; i < cudaDeviceCount; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		fprintf(stdout, "device_%d: %s\n", i, prop.name);
	}

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one input/output).
    cudaStatus = cudaMalloc((void**)&dev_image, imageSize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_descriptors, descriptorStep*4 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_weakClassProb, weakClassnum * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_numOfThreadInImage, descriptorStep * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_image, image, imageSize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_descriptors, descriptors, descriptorStep*4 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_weakClassProb, weakClassProb, weakClassnum * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_numOfThreadInImage, threadsOnImage, descriptorStep * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// ------- DATA ON GPU ----------

	int warpNumber = numberOfThread / WARP_SIZE;

	fprintf(stdout, "Number of threads: %d, Number of warps: %d, Warp size: %d\n", numberOfThread, warpNumber, WARP_SIZE);

    // Launch a kernel on the GPU with one thread for each element.
    applyClassifiersKernel<<<warpNumber+WARP_SIZE-1, WARP_SIZE>>>(dev_image, dev_descriptors, dev_weakClassProb, descriptorStep, imageSize, dev_numOfThreadInImage, weakClassnum);

	// ------- COMPUTATION DONE --------

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(image, dev_image, imageSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	cudaFree(dev_image);
    cudaFree(dev_descriptors);
    cudaFree(dev_weakClassProb);
	cudaFree(dev_numOfThreadInImage);
    
    return cudaStatus;
}
