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
 * @date      05 November  2020, 17:40 (created) \n
 * @date      05 November  2020, 18:35 (revised) \n
 *
 */

#include "wb.h"

/**
 * Compute reference output.
 * @param [in]  input - Input data
 * @param [in]  size  - Size of the array
 */
int compute(int* input,
            int  size)
{
  int ret = 0;
  for (int i = 0; i < size; i++)
  {
    ret += input[i];
  }
  return ret;
}// end of compute
//----------------------------------------------------------------------------------------------------------------------

/**
 * Allocate and generate data
 * @param [int] size - Size of the array.
 * @return generated array
 */
int* generate_data(int size)
{
  int* data = (int *) malloc(sizeof(int) * size);

  for (int i = 0; i < size; i++)
  {
    data[i] = (rand() % 20) - 10;
  }
  return data;
}// end of generate_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write string into a file.
 * @param [in] file_name - File Name
 * @param [in] data      - Data array
 * @param [in] size      - Size of the data array
 */
void write_data(const char* file_name,
                const int*  data,
                const int   size)
{
  FILE* handle = fopen(file_name, "w");

  fprintf(handle, "%d", size);
  for (int i = 0; i < size; i++)
  {
    fprintf(handle, "\n%d", data[i]);
  }
  fflush(handle);
  fclose(handle);
}// end of write_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create a dataset from a predefined string
 * @param [in] base_dir   - Base directory
 * @param [in] datasetNum - Dataset number
 * @param [in] size       - String to write
 */
void create_dataset(const char* base_dir,
                    const int   datasetNum,
                    const int   size)
{
  const char *dir_name = wbDirectory_create(wbPath_join(base_dir, datasetNum));

  char* input_file_name  = wbPath_join(dir_name, "input.raw");
  char* output_file_name = wbPath_join(dir_name, "output.raw");

  int* input_data = generate_data(size);
  int  output_data;

  output_data = compute(input_data, size);

  write_data(input_file_name, input_data, size);
  write_data(output_file_name, &output_data, 1);

  free(input_data);
}// end of create_dataset
//----------------------------------------------------------------------------------------------------------------------

/**
 * main routine
 * @return
 */
int main()
{
  const char* base_dir = wbPath_join(wbDirectory_current(), "Reduction", "Dataset");

  create_dataset(base_dir, 0, 64);
  create_dataset(base_dir, 1, 100);
  create_dataset(base_dir, 2, 1023);
  create_dataset(base_dir, 3, 2048);
  create_dataset(base_dir, 4, 1024 * 1024);      // 4MB
  create_dataset(base_dir, 5, 16 * 1024 * 1024); // 64MB

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
