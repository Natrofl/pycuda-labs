#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "wb.h"

int main(int argc, char *argv[]) {
  wbArg_t args;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  int inputLength;

  args = wbArg_read(argc, argv); /* чтение входных аргументов */

  wbTime_start(Generic, "Importing data to host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  wbTime_stop(Generic, "Importing data to host");

  hostOutput = (float *)malloc(sizeof(float) * inputLength);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");


  wbTime_start(GPU, "Doing GPU memory allocation");
  thrust::device_vector<float> deviceInput1(inputLength);
  thrust::device_vector<float> deviceInput2(inputLength);
  thrust::device_vector<float> deviceOutput(inputLength);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  thrust::copy(hostInput1, hostInput1 + inputLength, deviceInput1.begin());
  thrust::copy(hostInput2, hostInput2 + inputLength, deviceInput2.begin());
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  thrust::transform(deviceInput1.begin(), deviceInput1.end(),
                    deviceInput2.begin(), deviceOutput.begin(),
                    thrust::plus<float>());
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  thrust::copy(deviceOutput.begin(), deviceOutput.end(), hostOutput);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  return 0;
}
