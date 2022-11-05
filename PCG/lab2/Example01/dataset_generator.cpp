/**
 * @file      dataset_generator.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 2 / Matrix multiplication - naive version
 *
 * @version   2021
 *
 * @date      16 October   2020, 17:53 (created) \n
 * @date      21 October   2020, 10:08 (created) \n
 *
 */
#include "wb.h"

/**
 * Compute reference output.
 * @param [out] output - Output matrix
 * @param [in]  input0 - First input matrix
 * @param [in]  input1 - Second input matrix
 * @param N - Number of rows of the first matrix
 * @param M - Number of cols of the first matrix
 * @param P - Number of cols of the second matrix
 */
void compute(float*       output,
             const float* input0,
             const float* input1,
             const int    N,
             const int    M,
             const int    P)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < P; j++)
    {
      float sum = 0.f;
      #pragma omp simd reduction(+:sum)
      for (int k = 0; k < M; k++)
      {
        sum += input0[i * M + k] * input1[k * P + j];
      }
      output[i * P + j] = sum;
    }
  }
}// end of compute
//----------------------------------------------------------------------------------------------------------------------

/**
 * Allocate and generate input data.
 *
 * @param [in] height - Number of rows.
 * @param [in] width  - Number of cols.
 *
 * @return array filled with the data
 */
float* generate_data(int height, int width)
{
  float* data = (float *) malloc(sizeof(float) * width * height);

  for (int i = 0; i < width * height; i++)
  {
    data[i] = ((float)(rand() % 20) - 5) / 5.0f;
  }

  return data;
}// end of generate_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write data to file.
 * @param [in] file_name - File Name
 * @param [in] data      - Data array
 * @param [in] height    - Matrix height (number of rows)
 * @param [in] width     - Matrix width  (number of cols)
 */
void write_data(const char*  file_name,
                const float* data,
                const int    height,
                const int    width)
{
  FILE* handle = fopen(file_name, "w");

  fprintf(handle, "%d %d\n", height, width);

  for (int row = 0; row < height; row++)
  {
    for (int col = 0; col < width; col++)
    {
      fprintf(handle, "%.2f", *data++);
      if (col != width - 1)
      {
        fprintf(handle, " ");
      }
    }
    if (row != height - 1)
    {
      fprintf(handle, "\n");
    }
  }

  fflush(handle);
  fclose(handle);
}// end of write_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create necessary datasets
 * @param [in] datasetNum - Directory where to store data
 * @param [in] N          - Matrix [N * M]
 * @param [in] M          - Matrix [N * M] and Matrix [M * P]
 * @param [in] P          - Matrix [M * P]
 */
void create_dataset(const char* base_dir,
                    int         datasetNum,
                    int         N,
                    int         M,
                    int         P)
{
  const char* dir_name = wbDirectory_create(wbPath_join(base_dir, datasetNum));

  char* input0_file_name = wbPath_join(dir_name, "input0.raw");
  char* input1_file_name = wbPath_join(dir_name, "input1.raw");
  char* output_file_name = wbPath_join(dir_name, "output.raw");

  printf("Testing matrices [%d, %d] x [%d, %d]. \n", N, M, M, P);

  printf("  - Generating data...");
  fflush(stdout);
  float* input0_data = generate_data(N, M);
  float* input1_data = generate_data(M, P);
  float* output_data = (float *) calloc(sizeof(float), N * P);

  printf("Done \n");

  printf("  - Computing reference answer...");
  fflush(stdout);
  compute(output_data, input0_data, input1_data, N, M, P);
  printf("Done \n");

  printf("  - Writing data...");
  fflush(stdout);
  write_data(input0_file_name, input0_data, N, M);
  write_data(input1_file_name, input1_data, M, P);
  write_data(output_file_name, output_data, N, P);
  printf("Done \n");

  free(input0_data);
  free(input1_data);
  free(output_data);
} // end of create_dataset
//----------------------------------------------------------------------------------------------------------------------


/**
 * main routine
 * @return
 */
int main()
{

  const char* base_dir = wbPath_join(wbDirectory_current(), "MatrixMultiplication", "Dataset");

  //                [N * P] = [N * M] x [M * P]
  //                             N     M     P
  create_dataset(base_dir, 0,    4,    2,    4);
  create_dataset(base_dir, 1,  256,  256,  256);
  create_dataset(base_dir, 2,  100,  200,  300);
  create_dataset(base_dir, 3, 1024, 1024, 1024);
  create_dataset(base_dir, 4, 2048, 2048, 2048);
  create_dataset(base_dir, 5, 3072, 3072, 3072);

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
