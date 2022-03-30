#include <iostream>
#include <string>
#include "defs.h"
#include "Eigen/Core"

//HandlingEigenMatrix
void Sample1_EMatHandle()
{
    //RawData
    unsigned char raw_size = 4;
    size_t mem_size = raw_size * sizeof(float);
    float* src = new float[raw_size];
    float* dst = new float[raw_size];
    memset(src, 0, mem_size);
    memset(dst, 0, mem_size);
    float cur = 1.0f;
    for (size_t i = 0; i < raw_size; i++)
    {
        src[i] = cur;
        cur += 1.0;
    }


    //HandleData
    EMatXf e_src = EMatXf::Zero(2, 2);
    EMatXf e_dst = EMatXf::Zero(2, 2);

    std::cout << "InitialMat" << std::endl;
    std::cout << e_src << std::endl;

    Arr2EMat(e_src, src, raw_size);

    std::cout << "UpdatedMat" << std::endl;
    std::cout << e_src << std::endl;

    EMatOperate(e_dst, e_src);

    std::cout << "ProcedMat" << std::endl;
    std::cout << e_dst << std::endl;

    EMat2Arr(dst, raw_size, e_src);

    std::cout << "ResArr" << std::endl;
    for (size_t i = 0; i < raw_size; i++)
    {
        std::cout << dst[i] << ",";
    }
    std::cout << std::endl;

    //Mem Release
    e_src.resize(0, 0);
    e_dst.resize(0, 0);
    delete[] src;
    delete[] dst;
}

//Test 2D-UnitalyDCT for SquareInput
void Sample2_Uni2DDCT_Sq()
{
    
}

//MainProc
int main()
{
    Sample1_EMatHandle();
    return 0;
}
