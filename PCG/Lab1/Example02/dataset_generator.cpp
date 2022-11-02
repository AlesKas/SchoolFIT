/**
 * @file      dataset_generator.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 1 / Vector addition
 *
 * @version   2021
 *
 * @date      07 October   2020, 22:13 (created) \n
 * @date      07 October   2020, 22:13 (revised) \n
 *
 */

#include <wb.h>

/**
 * Compute reference output.
 * @param [out] output - Output array
 * @param [in]  input0 - first input array
 * @param [in]  input1 - second input array
 * @param [in]  size   - number of elements.
 */
void compute(float*       output,
             const float* input0,
             const float* input1,
             const int    size)
{
  for (int i = 0; i < size; i++)
  {
    output[i] = input0[i] + input1[i];
  }
}// end of compute
//----------------------------------------------------------------------------------------------------------------------

/**
 * Allocate and generate input data.
 *
 * @param  [in] size - Size of the array.
 * @return array filled with the data
 */
float* generate_data(int size)
{
  float *data = (float*) malloc(sizeof(float) * size);

  for (int i = 0; i < size; i++) {
    data[i] = ((float)(rand() % 20) - 5) / 5.0f;
  }

  return data;
}// end of generate_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write data to file.
 * @param [in] file_name - File name
 * @param [in] data      - Data array
 * @param [in] size      - array size
 */
void write_data(const char*  file_name,
                const float* data,
                const int    size)
{
  FILE* handle = fopen(file_name, "w");

  fprintf(handle, "%d", size);

  for (int i = 0; i < size; i++)
  {
    fprintf(handle, "\n%.2f", *data++);
  }
  fflush(handle);

  fclose(handle);
}// end of write_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create necessary datasets
 * @param [in] base_dir   - Directory where to store data
 * @param [in] datasetNum - Datasets number
 * @param [in] size       - Dataset size
 */
void create_dataset(const char* base_dir,
                    int         datasetNum,
                    int         size)
{
  const char* dir_name = wbDirectory_create(wbPath_join(base_dir, datasetNum));

  const char* input0_file_name = wbPath_join(dir_name, "input0.raw");
  const char* input1_file_name = wbPath_join(dir_name, "input1.raw");
  const char* output_file_name = wbPath_join(dir_name, "output.raw");

  printf("Testing data with %d elements. \n", size);

  printf("  - Generating data...");
  fflush(stdout);
  float* input0_data = generate_data(size);
  float* input1_data = generate_data(size);
  float* output_data = (float*) calloc(sizeof(float), size);
  printf("Done \n");

  printf("  - Computing reference answer...");
  fflush(stdout);
  compute(output_data, input0_data, input1_data, size);
  printf("Done \n");

  printf("  - Writing data...");
  fflush(stdout);
  write_data(input0_file_name, input0_data, size);
  write_data(input1_file_name, input1_data, size);
  write_data(output_file_name, output_data, size);
  printf("Done \n");

  free(input0_data);
  free(input1_data);
  free(output_data);
}// end of create_dataset
//----------------------------------------------------------------------------------------------------------------------

/**
 * main routine
 * @return
 */
int main() {
  const char* base_dir = wbPath_join(wbDirectory_current(), "VectorAdd", "Dataset");

  create_dataset(base_dir, 0, 16);                 // 16 elements (less than one warp)
  create_dataset(base_dir, 1, 64);                 // 64 elements ( four warps)
  create_dataset(base_dir, 2, 63);                 // 63 elements - an odd number
  create_dataset(base_dir, 3, 2000);               // two CUDA blocks
  create_dataset(base_dir, 4, 65536);              // 256KB per array
  create_dataset(base_dir, 5, 1024 * 1024);        // 4MB per array
  create_dataset(base_dir, 6, 16 * 1024 * 1024);   // 64MB per array

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
