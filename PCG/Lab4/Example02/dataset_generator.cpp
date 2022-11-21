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
#include <string.h>

using std::string;

// Number of bins in the histogram
const size_t NUM_BINS = 128;

/**
 * Compute reference output.
 * @param [out] histogram - computed histogram
 * @param [in]  numBins   - Number of bins
 * @param [in]  input     - Input data
 */
void compute(unsigned int* histogram,
             const int     numBins,
             const string& input)
{
  for (int i = 0; i < input.size(); i++)
  {
    // if the input character is out of range, use the last bin
    int bin = std::min(int(input[i]), numBins - 1);
    histogram[bin]++;
  }
}// end  of compute
//----------------------------------------------------------------------------------------------------------------------

/**
 * Allocate and generate string data
 * @param [int] size - Size of the string.
 * @return String
 */
std::string generate_data(size_t size)
{
  string str = "";
  for ( int i = 0; i < size; i++)
  {
    str+= (rand() % (NUM_BINS - 32)) + 32; // random printable character
  }
  return str;
}// end of generate_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write string into a file.
 * @param [in] file_name - File Name
 * @param [in] str       - Data array
 */
void write_data(const char*   file_name,
                const string& str)
{
  FILE* handle = fopen(file_name, "w");

  fprintf(handle, "%s", str.c_str());

  fflush(handle);
  fclose(handle);
}// end of write_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write histogram into a file.
 * @param file_name
 * @param data
 * @param num
 */
void write_data(const char*         file_name,
                const unsigned int* histogram,
                const int           size)
{
  FILE *handle = fopen(file_name, "w");

  fprintf(handle, "%d", size);

  for (int i = 0; i < size; i++) {
    fprintf(handle, "\n%d", histogram[i]);
  }

  fflush(handle);
  fclose(handle);
}// end of write_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create a dataset from a predefined string
 * @param [in] base_dir   - Base directory
 * @param [in] datasetNum - Dataset number
 * @param [in] str        - String to write
 */
void create_dataset_fixed(const char*   base_dir,
                          const int     datasetNum,
                          const string& str)
{
  const char* dir_name = wbDirectory_create(wbPath_join(base_dir, datasetNum));

  char* input_file_name  = wbPath_join(dir_name, "input.txt");
  char* output_file_name = wbPath_join(dir_name, "output.raw");

  unsigned int* output_data = (unsigned int *)calloc(NUM_BINS, sizeof(unsigned int));

  compute(output_data, NUM_BINS, str);

  write_data(input_file_name, str);
  write_data(output_file_name, output_data, NUM_BINS);

  free(output_data);
  free(input_file_name);
  free(output_file_name);
}// end of create_dataset_fixed
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create a dataset from random datas
 * @param [in[ base_dir     - Base directory
 * @param [in[ datasetNum   - Dataset number
 * @param [in[ input_length - size of the input string
 */
static void create_dataset_random(const char*  base_dir,
                                  const int    datasetNum,
                                  const size_t input_length)
{
  const char *dir_name   = wbDirectory_create(wbPath_join(base_dir, datasetNum));

  char *input_file_name  = wbPath_join(dir_name, "input.txt");
  char *output_file_name = wbPath_join(dir_name, "output.raw");

  std::string str = generate_data(input_length);
  unsigned int *output_data = (unsigned int *) calloc(NUM_BINS, sizeof(unsigned int));

  compute(output_data, NUM_BINS, str);

  write_data(input_file_name, str);
  write_data(output_file_name, output_data, NUM_BINS);

  free(output_data);
  free(input_file_name);
  free(output_file_name);
}// end of create_dataset_random
//----------------------------------------------------------------------------------------------------------------------

/**
 * main routine
 * @return
 */
int main()
{
  const char* base_dir = wbPath_join(wbDirectory_current(), "TextHistogram", "Dataset");

  create_dataset_fixed(base_dir,  0, "the quick brown fox jumps over the lazy dog");
  create_dataset_fixed(base_dir,  1, "gpu teaching kit - accelerated computing");
  create_dataset_random(base_dir, 2, 16);
  create_dataset_random(base_dir, 3, 2000);
  create_dataset_random(base_dir, 4, 1024 * 1024);      // 1MB
  create_dataset_random(base_dir, 5, 64 * 1024 * 1024); // 64MB

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
