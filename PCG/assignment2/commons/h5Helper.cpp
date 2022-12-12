/**
 * @file      h5Helper.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PCG - Assignment 2
 *            HDF5 helper routines
 *
 * @version   2022
 *
 * @date      11 November  2020, 11:22 (created) \n
 * @date      02 December  2020, 16:44 (revised) \n
 *
 */

#include "h5Helper.h"


using std::string;
//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Initialization ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Particle names initialization
std::vector<std::string>H5Helper::mParticleAttributeNames =
{
  "pos_x",
  "pos_y",
  "pos_z",
  "vel_x",
  "vel_y",
  "vel_z",
  "weight"
};// end of mParticleAttributeNames
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor
 */
H5Helper::~H5Helper()
{
  if (mInputFile !=  H5I_BADID)
  {
    H5Fclose(mInputFile);
  }
  if(mOutputFile > H5I_BADID)
  {
    H5Fclose(mOutputFile);
  }
}// end of destructor
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialize H5Helper
 */
void H5Helper::init()
{
  //---------------------------------- Open input file and read number of particles ----------------------------------//
  // Open input file
  mInputFile  = H5Fopen(mInputFileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // If cannot open a file, throw an error
  if (mInputFile < 0)
  {
    throw std::runtime_error("Could not open file!");
  }

  // Read dataset dimensions - number of particles
  hid_t dataset   = H5Dopen(mInputFile, mParticleAttributeNames[0].c_str(), H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);

  const int ndims = H5Sget_simple_extent_ndims(dataspace);

  hsize_t dims[ndims];
  H5Sget_simple_extent_dims(dataspace, dims, nullptr);

  /// Find out number of particles
  mNumberOfParticles = 1;
  for(int i = 0; i < ndims; i++)
  {
    mNumberOfParticles *= dims[i];
  }

  if(mNumberOfParticles < mMemoryDescriptor.mSize)
  {
    throw std::runtime_error("Input file contains less elements than required!");
  }
  else
  {
    if(mNumberOfParticles > mMemoryDescriptor.mSize)
    {
      std::cout<<"Input file contains more elements than required! Ommiting the rest."<<std::endl;
    }
  }
  H5Sclose(dataspace);
  H5Dclose(dataset);

  //------------------------------------- Create output file and all datasets ----------------------------------------//
  mOutputFile = H5Fcreate(mOutputFileName.c_str(), H5F_ACC_TRUNC,  H5P_DEFAULT, H5P_DEFAULT);

  // Lambda to create a dataset
  auto createDataset = [](hid_t& file, const string& datasetName, const hid_t& dataspace)
  {
    return H5Dcreate(file, datasetName.c_str(), H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  };

  /// Create 2D datasets [Particles x timeSteps]
  if(mMemoryDescriptor.mRecordsNum > 0)
  {

    for(int attributeId = 0; attributeId < kNumberOfParticleAttributes; attributeId++)
    {
      hsize_t dims[2] = {mMemoryDescriptor.mSize, mMemoryDescriptor.mRecordsNum};
      hid_t dataspace = H5Screate_simple(2, dims, nullptr);
      hid_t dataset   = createDataset(mOutputFile, mParticleAttributeNames[attributeId], dataspace);

      H5Sclose(dataspace);
      H5Dclose(dataset);
    }
  }// end create datasets

  // Create final state of the simulation
  for(int attributeId = 0; attributeId < kNumberOfParticleAttributes; attributeId++)
  {
    std::string sufix = "_final";

    hsize_t dims    = mMemoryDescriptor.mSize;
    hid_t dataspace = H5Screate_simple(1, &dims, nullptr);
    hid_t dataset   = createDataset(mOutputFile, mParticleAttributeNames[attributeId] + sufix, dataspace);

    H5Sclose(dataspace);
    H5Dclose(dataset);
  } // end of create final state

  /// Center of mass for timesteps
  if(mMemoryDescriptor.mRecordsNum > 0)
  {
    hsize_t dim  = mMemoryDescriptor.mRecordsNum;
    dataspace = H5Screate_simple(1, &dim, nullptr);

    dataset   = createDataset(mOutputFile, "com_x", dataspace);
    H5Dclose(dataset);

    dataset   = createDataset(mOutputFile, "com_y", dataspace);
    H5Dclose(dataset);

    dataset   = createDataset(mOutputFile, "com_z", dataspace);
    H5Dclose(dataset);

    dataset   = createDataset(mOutputFile, "com_w", dataspace);
    H5Dclose(dataset);

    H5Sclose(dataspace);
  }

  /// Final center of mass
  hsize_t dim  = 1;
  dataspace = H5Screate_simple(1, &dim, nullptr);

  dataset   = createDataset(mOutputFile, "com_x_final", dataspace);
  H5Dclose(dataset);

  dataset   = createDataset(mOutputFile, "com_y_final", dataspace);
  H5Dclose(dataset);

  dataset   = createDataset(mOutputFile, "com_z_final", dataspace);
  H5Dclose(dataset);

  dataset   = createDataset(mOutputFile, "com_w_final", dataspace);
  H5Dclose(dataset);

  H5Sclose(dataspace);
}// end of init
//----------------------------------------------------------------------------------------------------------------------

/**
 * Read particles from the input file using the memory descriptor
 */
void H5Helper::readParticleData()
{
  for(int attributeId = 0; attributeId < kNumberOfParticleAttributes; attributeId++)
  {
    hid_t dataset  = H5Dopen(mInputFile, mParticleAttributeNames[attributeId].c_str(), H5P_DEFAULT);

    hid_t dataspace = H5Dget_space(dataset);
    hsize_t dStart  = 0;
    hsize_t dStride = 1;
    hsize_t dCount  = mMemoryDescriptor.mSize;
    hsize_t dBlock  = 1;

    // Select a hyperslab
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &dStart, &dStride, &dCount, &dBlock);

    hsize_t dim    = mMemoryDescriptor.mStride[attributeId] * mMemoryDescriptor.mSize;
    hid_t memspace = H5Screate_simple(1, &dim, nullptr);

    hsize_t start  = mMemoryDescriptor.mOffset[attributeId];
    hsize_t stride = mMemoryDescriptor.mStride[attributeId];
    hsize_t count  = mMemoryDescriptor.mSize;
    hsize_t block  = 1;

    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &start, &stride, &count, &block);

    H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, mMemoryDescriptor.mDataPtr[attributeId]);

    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
  }
}// end of readParticleData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write particles from a given iteration
 * @param [in] iteration - Number of the iteration (time step)
 */
void H5Helper::writeParticleData(const size_t iteration)
{

  for(int attributeId = 0; attributeId < kNumberOfParticleAttributes; attributeId++)
  {
    hid_t dataset  = H5Dopen(mOutputFile, mParticleAttributeNames[attributeId].c_str(), H5P_DEFAULT);

    hid_t dataspace = H5Dget_space(dataset);

    hsize_t dStart[2]  = {0, iteration};
    hsize_t dCount[2]  = {mMemoryDescriptor.mSize, 1};

    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, dStart, nullptr, dCount, nullptr);

    hsize_t dim    = mMemoryDescriptor.mStride[attributeId] * mMemoryDescriptor.mSize;
    hid_t memspace = H5Screate_simple(1, &dim, nullptr);

    hsize_t start  = mMemoryDescriptor.mOffset[attributeId];
    hsize_t stride = mMemoryDescriptor.mStride[attributeId];
    hsize_t count  = mMemoryDescriptor.mSize;
    hsize_t block  = 1;

    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &start, &stride, &count, &block);

    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, mMemoryDescriptor.mDataPtr[attributeId]);

    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
  }
}// end of writeParticleData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write final state of the simulation
 */
void H5Helper::writeParticleDataFinal()
{

  for(int attributeId = 0; attributeId < kNumberOfParticleAttributes; attributeId++)
  {
    string sufix = "_final";
    hid_t dataset  = H5Dopen(mOutputFile, (mParticleAttributeNames[attributeId] + sufix).c_str(), H5P_DEFAULT);

    hid_t dataspace = H5Dget_space(dataset);

    hsize_t dim = mMemoryDescriptor.mStride[attributeId] * mMemoryDescriptor.mSize;
    hid_t memspace = H5Screate_simple(1, &dim, nullptr);

    hsize_t start  = mMemoryDescriptor.mOffset[attributeId];
    hsize_t stride = mMemoryDescriptor.mStride[attributeId];
    hsize_t count  = mMemoryDescriptor.mSize;
    hsize_t block  = 1;

    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &start, &stride, &count, &block);

    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, H5S_ALL, H5P_DEFAULT, mMemoryDescriptor.mDataPtr[attributeId]);

    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
  }
}// end of writeParticleDataFinal
//----------------------------------------------------------------------------------------------------------------------

/// Write Center of Mass
void H5Helper::writeCom(const float  comX,
                        const float  comY,
                        const float  comZ,
                        const float  comW,
                        const size_t record)
{
  // Set up dimension sizes
  hid_t dataset   = H5Dopen(mOutputFile, "com_x", H5P_DEFAULT);
  hid_t dataspace = H5Dget_space(dataset);
  hsize_t coords  = record;
  hsize_t dim = 1;
  hid_t memspace = H5Screate_simple(1, &dim, nullptr);

  H5Sclose(dataspace);
  H5Dclose(dataset);


  auto writeComAttribute = [this, &memspace, coords](const string& name, const float value)
  {
    hid_t dataset   = H5Dopen(mOutputFile, name.c_str(), H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset);

    H5Sselect_elements(dataspace, H5S_SELECT_SET, 1, &coords);

    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, &value);

    H5Sclose(dataspace);
    H5Dclose(dataset);
  }; // lambda to write an attribute

  writeComAttribute("com_x", comX);
  writeComAttribute("com_y", comY);
  writeComAttribute("com_z", comZ);
  writeComAttribute("com_w", comW);
}// end of writeCom
//----------------------------------------------------------------------------------------------------------------------

/// Write final Center of Mass
void H5Helper::writeComFinal(const float comX,
                             const float comY,
                             const float comZ,
                             const float comW)
{
  // Lambda to write a dataset
  auto writeDataset = [this](const string& name, const float value)
  {
    hid_t dataset  = H5Dopen(mOutputFile, name.c_str(), H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    H5Dclose(dataset);
  };

  writeDataset("com_x_final" , comX);
  writeDataset("com_y_final" , comY);
  writeDataset("com_z_final" , comZ);
  writeDataset("com_w_final" , comW);
}// end of writeComFinal
//----------------------------------------------------------------------------------------------------------------------
