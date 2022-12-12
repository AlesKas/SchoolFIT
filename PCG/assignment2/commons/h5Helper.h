/**
 * @file      h5Helper.h
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

#ifndef __H5HELPER_H__
#define __H5HELPER_H__

#include <cstddef>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>

#include <hdf5_hl.h>
#include <hdf5.h>

/// Forward declaration of H5Helper class
class H5Helper;

/**
 * @enum ParticleAttribute
 * Names of datasets used in the HDF5 file
 */
enum ParticleAttribute
{
  /// Position in the x direction.
  kPosX   = 0,
  /// Position in the y direction.
  kPosY   = 1,
  /// Position in the z direction.
  kPosZ   = 2,
  /// Velocity in the x direction.
  kVelX   = 3,
  /// Velocity in the y direction.
  kVelY   = 4,
  /// Velocity in the z direction.
  kVelZ   = 5,
  /// Weight
  kWeight = 6,
};// end of ParticleAttribute
//----------------------------------------------------------------------------------------------------------------------

/// Number of datasets in the file
constexpr int kNumberOfParticleAttributes = kWeight + 1;

/**
 * @class MemDesc
 * @brief This class is a memory descriptor specifying how the data is stored in memory
 */
class MemDesc
{
  public:
    /// Default constructor in not allowed
    MemDesc() = delete;

    /**
     * Constructor of the memory descriptor
     * @param [in] pos_x         - Position in x direction
     * @param [in] pos_x_stride  - Position stride
     * @param [in] pos_x_offset  - Position offset
     *
     * @param [in] pos_y         - Position in y direction
     * @param [in] pos_y_stride  - Position stride
     * @param [in] pos_y_offset  - Position offset
     *
     * @param [in] pos_z         - Position in z direction
     * @param [in] pos_z_stride  - Position stride
     * @param [in] pos_z_offset  - Position offset
     *
     * @param [in] vel_x         - Velocity in x direction
     * @param [in] vel_x_stride  - Velocity stride
     * @param [in] vel_x_offset  - Velocity offset
     *
     * @param [in] vel_y         - Velocity in y direction
     * @param [in] vel_y_stride  - Velocity stride
     * @param [in] vel_y_offset  - Velocity offset
     *
     * @param [in] vel_z         - Velocity in z direction
     * @param [in] vel_z_stride  - Velocity stride
     * @param [in] vel_z_offset  - Velocity offset
     *
     * @param [in] weight        - Weights of the particles
     * @param [in] weight_stride - Weights stride
     * @param [in] weight_offset - Weights offset
     *
     * @param [in] N             - Number of particles
     * @param [in] steps         - Number of time steps
     */
    MemDesc(float* pos_x,  const size_t pos_x_stride,  const size_t pos_x_offset,
            float* pos_y,  const size_t pos_y_stride,  const size_t pos_y_offset,
            float* pos_z,  const size_t pos_z_stride,  const size_t pos_z_offset,
            float* vel_x,  const size_t vel_x_stride,  const size_t vel_x_offset,
            float* vel_y,  const size_t vel_y_stride,  const size_t vel_y_offset,
            float* vel_z,  const size_t vel_z_stride,  const size_t vel_z_offset,
            float* weight, const size_t weight_stride, const size_t weight_offset,
            const size_t N,
            const size_t steps)
      : mDataPtr(kNumberOfParticleAttributes),
        mStride(kNumberOfParticleAttributes),
        mOffset(kNumberOfParticleAttributes),
        mSize(N),
        mRecordsNum(steps)
      {
        mDataPtr[ParticleAttribute::kPosX] = pos_x;
        mStride[ParticleAttribute::kPosX]  = pos_x_stride;
        mOffset[ParticleAttribute::kPosX]  = pos_x_offset;

        mDataPtr[ParticleAttribute::kPosY] = pos_y;
        mStride[ParticleAttribute::kPosY]  = pos_y_stride;
        mOffset[ParticleAttribute::kPosY]  = pos_y_offset;

        mDataPtr[ParticleAttribute::kPosZ] = pos_z;
        mStride[ParticleAttribute::kPosZ]  = pos_z_stride;
        mOffset[ParticleAttribute::kPosZ]  = pos_z_offset;

        mDataPtr[ParticleAttribute::kVelX] = vel_x;
        mStride[ParticleAttribute::kVelX]  = vel_x_stride;
        mOffset[ParticleAttribute::kVelX]  = vel_x_offset;

        mDataPtr[ParticleAttribute::kVelY] = vel_y;
        mStride[ParticleAttribute::kVelY]  = vel_y_stride;
        mOffset[ParticleAttribute::kVelY]  = vel_y_offset;

        mDataPtr[ParticleAttribute::kVelZ] = vel_z;
        mStride[ParticleAttribute::kVelZ]  = vel_z_stride;
        mOffset[ParticleAttribute::kVelZ]  = vel_z_offset;

        mDataPtr[ParticleAttribute::kWeight] = weight;
        mStride[ParticleAttribute::kWeight]  = weight_stride;
        mOffset[ParticleAttribute::kWeight]  = weight_offset;
      }// end of


    /// Use default destructor
    ~MemDesc() = default;

    /// Getter for the i-th particle's position in X
    float& getPosX(size_t index)    {return getElement(ParticleAttribute::kPosX, index);  }
    /// Getter for the i-th particle's position in Y
    float& getPosY(size_t index)    {return getElement(ParticleAttribute::kPosY, index);  }
    /// Getter for the i-th particle's position in Z
    float& getPosZ(size_t index)    {return getElement(ParticleAttribute::kPosZ, index);  }

    /// Getter for the i-th particle's position in X
    float& getVelX(size_t index)    {return getElement(ParticleAttribute::kVelX, index);  }
    /// Getter for the i-th particle's position in Y
    float& getVelY(size_t index)    {return getElement(ParticleAttribute::kVelY, index);  }
    /// Getter for the i-th particle's position in Z
    float& getVelZ(size_t index)    {return getElement(ParticleAttribute::kVelZ, index);  }


    /// Getter for the i-th particle's weight
    float& getWeight(size_t index)  {return getElement(ParticleAttribute::kWeight, index); }

    /// Getter for the data size - number of particles
    size_t getDataSize()            { return mSize;       }
    /// Getter number of records - number of time steps
    size_t getRecordsNum()          { return mRecordsNum; }

  protected:
    /// Pointers to data arrays
    std::vector<float*> mDataPtr;
    /// Pointers to data strides
    std::vector<size_t> mStride;
    /// Pointers to data offsets
    std::vector<size_t> mOffset;

    /// Size of the data
    size_t mSize;
    /// Number of records
    size_t mRecordsNum;
 private:
  // H5Helper is a friend of MemDesc
  friend H5Helper;

  /**
   * Get element from array
   * @param [in] attribute - Particle attribute
   * @param [in] index     - Index of the particle
   * @return
   */
  float& getElement(ParticleAttribute attribute,
                    size_t            index)
  {
    return mDataPtr[attribute][index * mStride[attribute] + mOffset[attribute]];
  }

};// end of MemDesc
//----------------------------------------------------------------------------------------------------------------------


/**
 * @class H5Helper
 * @brief Class responsible for reading and writing HDF5 files
 */
class H5Helper
{
  public:
    /// Default constructor is not allowed
    H5Helper() = delete;

    /**
     * Constructor
     * @param [in] inputFile       - Input file name
     * @param [in] outputFile      - Output file name
     * @param [in] memoryDesriptor - Memory descriptor
     */
    H5Helper(const std::string& inputFileName,
             const std::string& outputFileNames,
             MemDesc            memoryDesriptor)
      : mMemoryDescriptor(memoryDesriptor),
        mInputFileName(inputFileName),
        mOutputFileName(outputFileNames),
        mInputFile(H5I_BADID),
        mOutputFile(H5I_BADID)
      {

      }


    /// Destructor
    ~H5Helper();

    /// Initialize H5helper
	  void init();
    /// Read particle data from the input file
    void readParticleData();
    /**
     * Write particle data from a given iteration
     * @param [in] iteration - Iteration number (time step)
     */
    void writeParticleData(const size_t iteration);
    /// Write final state of the simulation
    void writeParticleDataFinal();

    /**
     * Write Center of Mass
     * @param [in] comX      - X coordinate of the Center of Mass
     * @param [in] comY      - Y coordinate of the Center of Mass
     * @param [in] comZ      - Z coordinate of the Center of Mass
     * @param [in] comW      - Total wight of all particles
     * @param [in] record    - Iteration (time step)
     */
    void writeCom(const float  comX,
                  const float  comY,
                  const float  comZ,
                  const float  comW,
                  const size_t record);
    /**
     * Write final Center of Mass
     * @param comX
     * @param comY
     * @param comZ
     * @param comW
     */
    void writeComFinal(const float comX,
                       const float comY,
                       const float comZ,
                       const float comW);

  private:
    /// Memory descriptor
	  MemDesc mMemoryDescriptor;
    /// Input file name
    const std::string mInputFileName;
    /// Output file name
    const std::string mOutputFileName;

    /// HDF5 id for the input file
    hid_t mInputFile;
    /// HDF5 id for the output file
    hid_t mOutputFile;

    /// Input size - Number of particles
    size_t mNumberOfParticles;
    /// Particle attributes names
    static std::vector<std::string> mParticleAttributeNames;
};// end of H5Helper
//----------------------------------------------------------------------------------------------------------------------

#endif /* __H5HELPER_H__ */
