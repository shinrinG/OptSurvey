#pragma once
#include "Eigen/Core"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "math.h"

typedef Eigen::MatrixXf EMatXf;
typedef Eigen::VectorXf EVecXf;
typedef unsigned char uchar;

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

bool CMat2EMat(
    EMatXf& dst,
    const cv::Mat& src
)
{
    bool bret = true;
    bool bExec = false;
    bExec = ((src.cols==dst.cols()) && (src.rows==dst.rows())) ? true : false;
    if(bExec)
    {
        size_t cur_idx = 0;
        for (size_t c = 0; c < src.cols; c++)
        {
            for (size_t r = 0; r < src.rows; r++)
            {
                cur_idx = r * src.cols + c;
                dst(r, c) = (float)src.data[cur_idx];
            }
        }

#ifdef _DEBUG
        int y = 3;
        int x = 2;
        std::cout << dst(y, x) << std::endl;
        std::cout << (int)src.at<unsigned char>(y,x) << std::endl;
        std::cout << dst(x, y) << std::endl;
        std::cout << (int)src.at<unsigned char>(x, y) << std::endl;
#endif
    }
    else
    {
        bret = false;
    }
    return bret;
}


unsigned char scaleFunc(float fin)
{
    return (unsigned char)((fin + 255.0) / 2);
}

bool EMat2CMat(
    cv::Mat& dst,
    const EMatXf& src,
    bool bEnableScale
)
{
    bool bret = true;
    bool bExec = false;
    bExec = ((src.cols() == dst.cols) && (src.rows() == dst.rows)) ? true : false;
    if (bExec)
    {
        size_t cur_idx = 0;
        float mul = bEnableScale ? 255.0f : 1.0f;
        for (size_t c = 0; c < src.cols(); c++)
        {
            for (size_t r = 0; r < src.rows(); r++)
            {
                cur_idx = r * src.cols() + c;
                dst.data[cur_idx] = bEnableScale? scaleFunc(src(r, c)):(unsigned char)(src(r,c));
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
        for(int kx =0;kx<sqsize;kx++)
        {
            for (int ky = 0; ky < sqsize; ky++)
            {
                //Gen 1 Basis
                EMatXf cur_img(sqsize, sqsize);
                for (int c = 0; c < sqsize; c++)
                {
                    for (int r = 0; r < sqsize; r++)
                    {
                        cur_img(r, c) = (float)(std::cos((2 * r + 1) * kx * pi / (2 * sqsize)) * std::cos((2 * c + 1) * ky * pi / (2 * sqsize)));
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
    const double pi = 3.141592653589793238463;
    size_t rows = src.rows();
    size_t cols = src.cols();

    //Judge Executable
    if((rows==dst.rows())&&(cols==dst.cols())&&(cols==rows)&&(cols != 0))
    {
        bExec = true;
    }

    //Execute
    if(bExec)
    {
        EMatXf e_conv = Eigen::MatrixXf::Zero(rows,cols);
        float coef = 0.0;
        for (size_t c = 0; c < cols; c++)
        {
            for (size_t r = 0; r < rows; r++)
            {
                coef = (r == 0) ? 1.0 : 2.0;
                e_conv(r, c) = std::sqrt(coef) * (float)std::cos((2 * c + 1) * r * pi / (2 * rows)) / std::sqrt(rows);
            }
        }
        EMatXf e_conv_t = e_conv.transpose();
        dst = e_conv * src * e_conv_t; //UnitaryTransform for Signal Matrix(Z = RXR^T)
    }
    else
    {
        bret = false;
    }

    return bret;
}


int objectiveFunc1(float res, const EVecXf& src)
{
    int ret = 0;
    size_t dim = 2;
    if(src.rows()==2)
    {
        res = 4 * std::pow(src(0)-2,2) + 3 * std::pow(src(1)-9,2);
    }
    else
    {
        res = -1.0f;
        ret = 1;
    }
    return ret;
}

bool LineSearch(
    EVecXf& res,
    const size_t dim,
    const EVecXf& norm_direction,
    int(*ofunc(float,const EVecXf&)),
    const float eps = 0.0001f
)
{
    bool bret = false;
    EVecXf cur_x = EVecXf::Zero(dim);

    return bret;
}