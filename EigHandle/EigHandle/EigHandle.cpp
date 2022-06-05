#include <iostream>
#include <string>
#include "defs.h"
#include "Eigen/Core"
#include <opencv2/opencv.hpp>
#include "sample31_deblur.h"

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

//Test Generate DictionaryImage 2D-UnitalyDCT
void Sample2_GenUni2DDCT_Sq()
{
    std::string INPUT_FILE = "";
    std::string OUTPUT_FOLDER = "D:\\test\\dctRes\\dict\\cos64";
    size_t basis_size = 64;
    //cv::Mat src = cv::imread(INPUT_FILE,CV_8U);
    //int src_w = src.cols;
    //int src_h = src.rows;

    DCTSqDictionary* cdict = new DCTSqDictionary(basis_size);

    cv::Mat tmp = cv::Mat::zeros(cv::Size(basis_size,basis_size),CV_8U);

    int idx_dict = 0;
    int idx_img = 0;
    std::string sname = "";
    float cur_val = 0.0f;
    unsigned char v = 0;
    for(int kx = 0; kx < basis_size; kx++)
    {
        for (int ky = 0; ky < basis_size; ky++)
        {
            idx_dict = basis_size * kx + ky;
            std::cout << idx_dict << std::endl;
            sname = OUTPUT_FOLDER + "\\Basis_Cos_" + std::to_string(ky) + "_" + std::to_string(kx) + ".bmp";
            for (int c = 0; c < basis_size; c++)
            {
                for (int r = 0; r < basis_size; r++)
                {
                    idx_img = c * basis_size + r;
                    cur_val = cdict->m_vterm.at(idx_dict)(r, c) + 1.0;
                    tmp.data[idx_img] = (unsigned char)(127 * cur_val);
                }
            }
            cv::imwrite(sname, tmp);
        }
    }

    delete cdict;
}

//Test Proc 2D-UnitalyDCT
void Sample2_Uni2DDCT_Sq()
{
    std::string fpath = "D:\\test\\SW20220403\\exp\\20220404005029\\tmp\\crop\\crp_0000.jpg";
    cv::Mat in = cv::imread(fpath, CV_8U);
    EMatXf e_in = Eigen::MatrixXf::Zero(in.rows, in.cols);
    CMat2EMat(e_in, in);
    
    EMatXf e_out = Eigen::MatrixXf::Zero(in.rows, in.cols);
    UnitaryDCT2DSq(e_out,e_in);

    cv::Mat out = cv::Mat::zeros(cv::Size(in.rows, in.cols), CV_8U);
    EMat2CMat(out,e_out,true);
    fpath = "D:\\test\\SW20220403\\exp\\20220404005029\\tmp\\glcm\\crp_0000_dct.bmp";
    cv::imwrite(fpath, out);
}

//Test Op Conjugate Gradient Method
void Sample3_CGMethod()
{
    size_t dim = 2;
    //Deplicate
}

//SampleForSModeling is S3X

void call_sample31_deblur()
{

}

//MainProc
int main()
{
    call_sample31_deblur();
    return 0;
}
