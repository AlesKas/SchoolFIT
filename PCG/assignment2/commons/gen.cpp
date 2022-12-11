/**
 * @file      gen.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PCG - Assignment 2
 *            Generate input data
 *
 * @version   2022
 *
 * @date      11 November  2020, 11:22 (created) \n
 * @date      02 December  2020, 16:44 (revised) \n
 *
 */

#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <ctime>
#include <vector>
#include <hdf5_hl.h>
#include <hdf5.h>

/**
 * Get a float random number
 * @return Random number without 0.
 */
float randf()
{
  const float a = float(rand()) / float(RAND_MAX);

  return (a == 0.0f) ? FLT_MIN : a;
}// end of randf
//----------------------------------------------------------------------------------------------------------------------

/**
 * Main routine. Generates N particles into an output file.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  srand(time(NULL));

  // Test command line parameters.
  if (argc != 3)
  {
    printf("Usage: gen <Number of particles> <output file>\n");
    exit(EXIT_FAILURE);
  }

  const size_t N = static_cast<size_t>(atoi(argv[1]));

  std::vector<float> pos_x(N);
  std::vector<float> pos_y(N);
  std::vector<float> pos_z(N);
  std::vector<float> vel_x(N);
  std::vector<float> vel_y(N);
  std::vector<float> vel_z(N);
  std::vector<float> weight(N);

  // Print parameters
  printf("Number of particles: %d\n", N);

  // Create HDF5 file
  hid_t hdf5File = H5Fcreate(argv[2], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (hdf5File == 0)
  {
    printf("Can't open file %s!\n", argv[2]);
    exit(EXIT_FAILURE);
  }

  // Create HDF5 file space
  hsize_t datasetSize = N;
  hid_t dataspace = H5Screate_simple(1, &datasetSize, nullptr);

  // Generate particles
  for (size_t i = 0; i < N; i++)
  {
    pos_x[i]  = randf() * 100.0f;
    pos_y[i]  = randf() * 100.0f;
    pos_z[i]  = randf() * 100.0f;

    vel_x[i]  = randf() * 4.0f - 2.0f;
    vel_y[i]  = randf() * 4.0f - 2.0f;
    vel_z[i]  = randf() * 4.0f - 2.0f;

    weight[i] = randf() * 2500000000.0f;
  }

  // Lambda to write dataset into a file
  auto write = [&hdf5File, &dataspace](const char* datasetName, const float* data)
  {
    hid_t dataset = H5Dcreate2(hdf5File, datasetName, H5T_NATIVE_FLOAT, dataspace,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dataset);
  };

  // Write data.
  write("pos_x", pos_x.data());
  write("pos_y", pos_y.data());
  write("pos_z", pos_z.data());

  write("vel_x", vel_x.data());
  write("vel_y", vel_y.data());
  write("vel_z", vel_z.data());

  write("weight", weight.data());


  // Clean up
  H5Sclose(dataspace);
  H5Fclose(hdf5File);

  return EXIT_FAILURE;
}// end of main
//----------------------------------------------------------------------------------------------------------------------