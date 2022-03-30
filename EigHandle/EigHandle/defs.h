#pragma once
#include "Eigen/Core"
#include <vector>
#include <string>
#include "math.h"

typedef Eigen::MatrixXf EMatXf;
typedef Eigen::VectorXf EVecXf;

bool Arr2EMat(
    EMatXf& dst,
    const float* const src,
    const size_t src_size
)
{
    bool bret = true;
    bool bExec = false;
    size_t rows = dst.rows();
    size_t cols = dst.cols();
    bExec = (rows * cols >= src_size) ? true : false;

    if (bExec)
    {
        size_t cur_num = 0;
        for (size_t c = 0; c < cols; c++)
        {
            for (size_t r = 0; r < rows; r++)
            {
                cur_num = c * rows + r;
                dst(r, c) = (cur_num < src_size) ? src[cur_num] : 0.0f;
            }
        }
    }
    else
    {
        bret = false;
    }

    return bret;
}

bool EMat2Arr(
    float* dst,
    size_t dst_size,
    const EMatXf& src
)
{
    bool bret = true;
    bool bExec = false;
    size_t rows = src.rows();
    size_t cols = src.cols();
    bExec = (rows * cols == dst_size) ? true : false;
    if (bExec)
    {
        size_t cur_num = 0;
        for (size_t c = 0; c < cols; c++)
        {
            for (size_t r = 0; r < rows; r++)
            {
                cur_num = c * rows + r;
                dst[cur_num] = src(r, c);
            }
        }
    }
    else
    {
        bret = false;
    }
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

/// <summary>
/// Dictionary of 2D-DCT Only Sqare
/// </summary>
class DCTSqDictionary{
public:
    size_t m_size;
    std::vector<EMatXf> m_vterm;// Vector of BasisImg
    std::vector<EVecXf> m_vbasis;// Vector of Basis

    DCTSqDictionary():
        m_size(1),
        m_vterm(0),
        m_vbasis(0)
    {
    }

    DCTSqDictionary(size_t sqsize):
        m_size(sqsize*sqsize),
        m_vterm(0)
    {
        const double pi = 3.141592653589793238463;
        // Dim^4
        for(int kx =0;kx<m_size;kx++)
        {
            for (int ky = 0; ky < m_size; ky++)
            {
                //Gen 1 Basis
                EMatXf cur_img(m_size, m_size);
                for (int c = 0; c < m_size; c++)
                {
                    for (int r = 0; r < m_size; r++)
                    {
                        cur_img(r, c) = (float)(std::cos((2 * c + 1) * kx * pi / (2 * m_size)) * std::cos((2 * r + 1) * ky * pi / (2 * m_size)));
                    }
                }
                m_vterm.push_back(cur_img);
            }

        }
        
    }

    ~DCTSqDictionary()
    {
        m_vterm.clear();
        m_vbasis.clear();
    }

    //TODO
    bool genBasis()
    {
        return false;
    }


};

/// <summary>
/// 2D-UnitaryDCT Only Sqare
/// </summary>
/// <param name="dst"></param>
/// <param name="src"></param>
/// <returns></returns>
bool UnitaryDCT2DSq(EMatXf& dst,const EMatXf& src)
{
    bool bret = true;
    bool bExec = false;
    size_t rows = src.rows();
    size_t cols = src.cols();
    if((rows==dst.rows())&&(cols==dst.cols())&&(cols==rows))
    {
        bExec = true;
    }

    if(bExec)
    {
        EMatXf e_conv = Eigen::MatrixXf::Zero(rows,cols);
        //TODO



        dst = e_conv * src * e_conv.transpose(); //UnitaryTransform for Signal Matrix(Z = RXR^T)
    }
    

    return bret;
}