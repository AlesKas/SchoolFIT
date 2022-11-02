/**
 * @file      dataset_generator.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PC lab 1 / Covert RGB image to levels of gray
 *
 * @version   2021
 *
 * @date      08 October   2020, 19:03 (created) \n
 * @date      08 October   2020, 19:47 (revised) \n
 *
 */

#include <wb.h>

using Pixel = unsigned char;

/**
 * Compute reference output.
 * @param [out] output - Output image
 * @param [in]  input  - Input image
 * @param [in]  height - Number of rows (y size)
 * @param [in]  width  - Number of cols (x size)
 */
void compute(Pixel*       output,
             const Pixel* input,
             const int    height,
             const int    width)
{
  for (unsigned int y = 0; y < height; y++)
  {
    for (unsigned int x = 0; x < width; x++)
    {
      unsigned int idx = y * width + x;

      const float r     = input[3 * idx];     // red value for pixel
      const float g     = input[3 * idx + 1]; // green value for pixel
      const float b     = input[3 * idx + 2]; // blue value for pixel

      // Convert to gray
      output[idx] = Pixel(0.21f * r + 0.71f * g + 0.07f * b);
    }
  }
}// end of compute
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate random images from noise
 * @param [in] height - Number of rows (y size)
 * @param [in] width  - Number of cols (x size)
 * @return array with the generated image.
 */
Pixel* generate_data(const int height,
                     const int width)
{
  /* raster of y rows
     R, then G, then B pixel
     if maxVal < 256, each channel is 1 byte
     else, each channel is 2 bytes
  */

  // Max RGB value
  constexpr int maxVal = 255;

  Pixel* image = (Pixel *) malloc (width * height * 3);

  for (int i = 0; i < width * height; i += 3)
  {
    image[i]     = rand() % maxVal;  // R
    image[i + 1] = rand() % maxVal;  // G
    image[i + 2] = rand() % maxVal;  // B
  }

  return image;
}// end of generate_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write data into the file.
 * @param [in] file_name - File name
 * @param [in] image     - Image to write
 * @param [in] height    - Height of the image
 * @param [in] width     - Width of the image
 * @param [in] channels  - Number of color channels
 */
void write_data(const char*  file_name,
                const Pixel* image,
                const int    height,
                const int    width,
                const int    channels)
{
  FILE *handle = fopen(file_name, "w");

  //
  const char* file_format = (channels == 1) ? "P5\n" : "P6\n";

  fprintf(handle, file_format);

  fprintf(handle, "#Created by %s\n", __FILE__);
  fprintf(handle, "%d %d\n", width, height);
  fprintf(handle, "255\n");

  fwrite(image, width * channels * sizeof(Pixel), height, handle);

  fflush(handle);
  fclose(handle);
}// end of write_data
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create datasets.
 * @param [in] base_dir - Directory where to store data
 * @param [in] height   - Height of the image
 * @param [in] width    - Width of the image
 * @param [in] size     - Dataset size
 */
void create_dataset(const char* base_dir,
                    const int   datasetNum,
                    const int   height,
                    const int   width)
{
  const char* dir_name = wbDirectory_create(wbPath_join(base_dir, datasetNum));

  const char* input_file_name  = wbPath_join(dir_name, "input.ppm");
  const char* output_file_name = wbPath_join(dir_name, "output.pbm");

  printf("Testing image with [%d, %d] pixels. \n", width, height);

  printf("  - Generating data...");
  fflush(stdout);
  Pixel *input_data  = generate_data(width, height);
  Pixel *output_data = (Pixel *) calloc(sizeof(Pixel), height * height  * 3);
  printf("Done \n");

  printf("  - Computing reference answer...");
  fflush(stdout);
  compute(output_data, input_data, height, width);
  printf("Done \n");

  printf("  - Writing data...");
  fflush(stdout);
  write_data(input_file_name,  input_data,  height, width, 3);
  write_data(output_file_name, output_data, height, width, 1);
  printf("Done \n");

  free(input_data);
  free(output_data);
}// end of create_dataset
//----------------------------------------------------------------------------------------------------------------------

/**
 * main routine
 * @return
 */
int main()
{
  const char* base_dir = wbPath_join(wbDirectory_current(), "ImageColorToGrayscale", "Dataset");

  create_dataset(base_dir, 0,  256,  256);
  create_dataset(base_dir, 1,  512,  256);
  create_dataset(base_dir, 2,  256,  512);
  create_dataset(base_dir, 3,  135,   79);
  create_dataset(base_dir, 4,   83,  127);
  create_dataset(base_dir, 5, 1080, 1920);

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
