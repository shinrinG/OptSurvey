#include <iostream>
#include "defs.h"
#include "Eigen/Core"

bool EigFunc(Eigen::MatrixXf& dst,const Eigen::MatrixXf& src)
{
    bool bret = true;
    


    return bret;
}

bool Arr2EMat() 
{
    bool bret = true;

    return bret;
}

bool EMat2Arr() 
{
    bool bret = true;

    return bret;
}

bool EMatOperate(Eigen::MatrixXf& dst, const Eigen::MatrixXf& src)
{
    bool bret = true;
    bret = (src.cols() == 2) && (dst.cols() == 2);
    if (bret) 
    {
        Eigen::MatrixXf base = Eigen::MatrixXf::Zero(2, 1);
        base(0, 0) = 1;
        base(1, 0) = 1;
        dst = src * base;
    }
    return bret;
}


//MainProc
int main()
{
    //RawData
    unsigned char raw_size = 4;
    size_t mem_size = raw_size * sizeof(float);
    float* src = new float[raw_size];
    memset(src, 0, mem_size);
    float cur = 1.0f;
    for (size_t i = 0; i < raw_size; i++)
    {
        src[i] = cur;
        cur+=1.0;
    }
    std::cout << src[3] << "\n";

    //HandleData
    EMatXf e_src = EMatXf::Zero(2,2);
    EMatXf e_dst = EMatXf::Zero(2, 2);

    
    std::cout << e_src;

    //Mem Release
    e_src.resize(0,0);
    e_dst.resize(0, 0);
    delete[] src;
    return 0;
}
