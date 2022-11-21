/**
 * @File gen.cpp
 *
 * Program pro generování vstupních dat.
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xlogin00
 */

#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <ctime>
#include <vector>
#include <hdf5_hl.h>
#include <hdf5.h>

/**
 * Generate float random numbers.
 * @return a ranom number
 */
float randf()
{
	float a = (float)rand() / (float)RAND_MAX;

	if (a == 0.0f)
  {
    a = FLT_MIN;
  }

	return a;
}// end of randf()
//----------------------------------------------------------------------------------------------------------------------


/**
 * main routine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv)
{
  srand(time(nullptr));

  // parse commandline parameters
  if (argc != 3)
  {
    printf("Usage: gen <N> <output>\n");
    exit(1);
  }

  const size_t N = static_cast<size_t>(atoi(argv[1]));
  // allocate memory
  std::vector<float> pos_x(N);
  std::vector<float> pos_y(N);
  std::vector<float> pos_z(N);
  std::vector<float> vel_x(N);
  std::vector<float> vel_y(N);
  std::vector<float> vel_z(N);
  std::vector<float> weight(N);

  // print parameters
  printf("N: %lu\n", N);

  // Create output file
  hid_t file_id  =  H5Fcreate(argv[2], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id == 0)
  {
    printf("Can't open file %s!\n", argv[2]);
    exit(1);
  }

  hsize_t size = N;
  hid_t dataspace = H5Screate_simple(1, &size, nullptr);

  // Generate random values
  for (int i = 0; i < N; i++)
  {
    pos_x[i] = randf() * 100.0f;
    pos_z[i] = randf() * 100.0f;
    pos_y[i] = randf() * 100.0f;
    vel_x[i] = randf() * 4.0f - 2.0f;
    vel_y[i] = randf() * 4.0f - 2.0f;
    vel_z[i] = randf() * 4.0f - 2.0f;
    weight[i] = randf() *  2500000000.0f;
  }

  // Lambda to write a dataset
  auto writeDataset = [=](const char* datasetName, std::vector<float>& vector)
  {
    hid_t dataset = H5Dcreate2(file_id, datasetName, H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vector.data());
    H5Dclose(dataset);
  };// end of v

  // write necessary datasets

  writeDataset("pos_x", pos_x);
  writeDataset("pos_y", pos_y);
  writeDataset("pos_z", pos_z);

  writeDataset("vel_x", vel_x);
  writeDataset("vel_y", vel_y);
  writeDataset("vel_z", vel_z);

  writeDataset("weight", weight);

  H5Sclose(dataspace);
  H5Fclose(file_id);

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------