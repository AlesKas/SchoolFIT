/**
 *
 * @File h5Helper.cpp
 *
 * Implementation file with routines for writing HDF5 files
 *
 * Paralelní programování na GPU (PCG 2022)
 * Projekt c. 1 (cuda)
 * Login: xlogin00
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

/// Forward definition for the H5Helper
class H5Helper;

/// Number of datasets
constexpr int kAttrNum = 7;

/// Enum for dataset names
enum Atr
{
  kPosX,
  kPosY,
  kPosZ,
  kVelX,
  kVelY,
  kVelZ,
  kWeight,
};

 /**
  * @class MemDesc
  * Memory descriptor class
  */
class MemDesc
{
  public:
    /**
     * Constructor
     * parameters:
     *                      Stride of two               Offset of the first
     *      Data pointer    consecutive elements        element in floats,
     *                      in floats, not bytes        not bytes
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
      : mDataPtr(kAttrNum), mStride(kAttrNum), mOffset(kAttrNum), mSize(N), mRecordsNum(steps)
      {
        mDataPtr[Atr::kPosX] = pos_x;
        mStride[Atr::kPosX]  = pos_x_stride;
        mOffset[Atr::kPosX]  = pos_x_offset;

        mDataPtr[Atr::kPosY] = pos_y;
        mStride[Atr::kPosY]  = pos_y_stride;
        mOffset[Atr::kPosY]  = pos_y_offset;

        mDataPtr[Atr::kPosZ] = pos_z;
        mStride[Atr::kPosZ]  = pos_z_stride;
        mOffset[Atr::kPosZ]  = pos_z_offset;

        mDataPtr[Atr::kVelX] = vel_x;
        mStride[Atr::kVelX]  = vel_x_stride;
        mOffset[Atr::kVelX]  = vel_x_offset;

        mDataPtr[Atr::kVelY] = vel_y;
        mStride[Atr::kVelY]  = vel_y_stride;
        mOffset[Atr::kVelY]  = vel_y_offset;

        mDataPtr[Atr::kVelZ] = vel_z;
        mStride[Atr::kVelZ]  = vel_z_stride;
        mOffset[Atr::kVelZ]  = vel_z_offset;

        mDataPtr[Atr::kWeight] = weight;
        mStride[Atr::kWeight]  = weight_stride;
        mOffset[Atr::kWeight]  = weight_offset;
      }

    /// Getter for the i-th particle's position in X
    float& getPosX(size_t i)
    {
      return mDataPtr[Atr::kPosX][i*mStride[Atr::kPosX] + mOffset[Atr::kPosX]];
    }

    /// Getter for the i-th particle's position in Y
    float& getPosY(size_t i)
    {
      return mDataPtr[Atr::kPosY][i*mStride[Atr::kPosY] + mOffset[Atr::kPosY]];
    }

    /// Getter for the i-th particle's position in Z
    float& getPosZ(size_t i)
    {
      return mDataPtr[Atr::kPosZ][i*mStride[Atr::kPosZ] + mOffset[Atr::kPosZ]];
    }

    /// Getter for the i-th particle's weight
    float& getWeight(size_t i)
    {
      return mDataPtr[Atr::kWeight][i*mStride[Atr::kWeight] + mOffset[Atr::kWeight]];
    }

    /// Getter for the data size
    size_t getDataSize(){ return mSize; }

    /// Default constructor is not allowed
    MemDesc() = delete;

  protected:
    /// Vector of data pointers
    std::vector<float*> mDataPtr;
    /// Stride of two consecutive elements in memory pointed to by data pointers (in floats, not bytes)
    std::vector<size_t> mStride;
    /// Offset of the first element in the memory pointed to by data pointers (in floats, not bytes)
    std::vector<size_t> mOffset;

    size_t mSize;
    size_t mRecordsNum;

  private:
    friend H5Helper;
};// end of MemDesc
//----------------------------------------------------------------------------------------------------------------------

/*

/**
 * @class H5Helper
 *
 * HDF5 file format reader and writer
 *
 * @param inputFile
 * @param outputFile
 * @param md
 */
class H5Helper
{
  public:

    /**
     * Constructor requires file names and memory layout descriptor
     * @param inputFile
     * @param outputFile
     * @param md - memory descriptor
     */
    H5Helper(const std::string& inputFile,
             const std::string& outputFile,
             MemDesc md)
    : mMd(md),
      mInputFile(inputFile), mOutputFile(outputFile),
      mInputFileId(0), mOutputFileId(0){}

    /// Default constructor is not allowed
    H5Helper() = delete;

    /// Destructor
    ~H5Helper();

    /// Initialize helper
    void init();

    /// Read input data to the memory according to the memory descriptor
    void readParticleData();
    /// Write simulation data to the output file according to the memory descriptor and record number
    void writeParticleData(const size_t record);
    /// Write final simulation data to the output file according to the memory descriptor
    void writeParticleDataFinal();
    /// Write center-of-mass to the output file according record number
    void writeCom(const float comX,
                  const float comY,
                  const float comZ,
                  const float comW,
                  const size_t record);
    /// Write final center-of-mass to the output file */
    void writeComFinal(const float comX,
                       const float comY,
                       const float comZ,
                       const float comW);
  private:
    /// Memory descriptor
   	MemDesc mMd;

    /// File names
    std::string mInputFile;
    std::string mOutputFile;

    /// File handles
    hid_t mInputFileId;
    hid_t mOutputFileId;

    /// Input size
    size_t mInputSize;

    /// Vector of dataset names
    static std::vector<std::string> mDatasetNames;
};// end of H5Helper
//----------------------------------------------------------------------------------------------------------------------

#endif /* __H5HELPER_H__ */
