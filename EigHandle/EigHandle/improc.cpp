#include "improc.h"


/**
************************************************************************************************************************
* @file improc.cpp
* @brief HaarWaveletTransForm-VerticalCore
* @param [in]
* @param [in]
* @param [in]
************************************************************************************************************************
**/
template<class T, class K>
bool wavelet_haar_core_vertical(K* p_dst_p, K* p_dst_n, const T* p_src, const size_t w,const size_t h)
{
	bool bret = true;
	size_t cur_idx = 0;
	size_t cur_idx_sub = 0;
	for(size_t y=0;y<h;y++)
	{
		for(size_t x=0;x<w-1;x++)//Roll Operation
		{
			cur_idx = y * w + x;
			p_dst_p[cur_idx] = (K)((p_src[cur_idx] + p_src[cur_idx + 1]) / 2);
			p_dst_n[cur_idx] = (K)((p_src[cur_idx] - p_src[cur_idx + 1]) / 2);
		}
		//Care of Right Col. Cyclic Connect.
		cur_idx = y * w + w - 1;
		cur_idx_sub = y * w;
		p_dst_p[cur_idx] = (K)((p_src[cur_idx] + p_src[cur_idx_sub]) / 2);
		p_dst_n[cur_idx] = (K)((p_src[cur_idx] - p_src[cur_idx_sub]) / 2);
	}
	return bret;
}

/**
************************************************************************************************************************
* @file improc.cpp
* @brief HaarWaveletTransForm-HorizanalCore
* @param [in]
* @param [in]
* @param [in]
************************************************************************************************************************
**/
template<class T, class K>
bool wavelet_haar_core_horizon(K* p_dst_p, K* p_dst_n, const T* p_src, const size_t w, const size_t h)
{
	bool bret = true;
	size_t cur_idx = 0;
	size_t cur_idx_sub = 0;
	for (size_t x = 0; x < w; x++)
	{
		for (size_t y = 0; y < h - 1; y++)//Roll Operation
		{
			cur_idx = y * w + x;
			cur_idx_sub = (y+1) * w + x;
			p_dst_p[cur_idx] = (K)((p_src[cur_idx] + p_src[cur_idx_sub]) / 2);
			p_dst_n[cur_idx] = (K)((p_src[cur_idx] - p_src[cur_idx_sub]) / 2);
		}
		//Care of Bottom Raw. Cyclic Connect.
		cur_idx = (h-1) * w + x;
		cur_idx_sub = x;
		p_dst_p[cur_idx] = (K)((p_src[cur_idx] + p_src[cur_idx_sub]) / 2);
		p_dst_n[cur_idx] = (K)((p_src[cur_idx] - p_src[cur_idx_sub]) / 2);
	}
	return bret;
}

/**
************************************************************************************************************************
* @file improc.cpp
* @brief HaarWaveletTransForm-Level1
* @param [in]
* @param [in]
* @param [in]
************************************************************************************************************************
**/
template<class T, class K>
bool wavelet_haar_lv1(K* p_dst_ll, K* p_dst_lh, K* p_dst_hl, K* p_dst_hh, const T* p_src, const size_t w, const size_t h)
{
	bool bret = true;
	size_t size_arr = w * h;
	size_t size_mem = size_arr * sizeof(K);
	K* p_tmp_l = new K[size_arr];
	K* p_tmp_h = new K[size_arr];
	memset(p_tmp_l, 0, size_mem);
	memset(p_tmp_h, 0, size_mem);

	//Transform of HaarWavelet
	wavelet_haar_core_horizon<T,K>(p_tmp_l,p_tmp_h,p_src,w,h);
	wavelet_haar_core_vertical<K, K>(p_dst_hl, p_dst_hh, p_tmp_h, w, h);
	wavelet_haar_core_vertical<K, K>(p_dst_ll, p_dst_lh, p_tmp_l, w, h);

	delete p_tmp_l;
	delete p_tmp_h;
	return bret;
}

/**
************************************************************************************************************************
* @file improc.cpp
* @brief HaarWaveletTransForm-Level2
* @param [in]
* @param [in]
* @param [in]
************************************************************************************************************************
**/
template<class T, class K>
bool wavelet_haar_lv2(K* p_dst_ll2, K* p_dst_lh2, K* p_dst_hl2, K* p_dst_hh2, K* p_dst_lh, K* p_dst_hl, K* p_dst_hh, const T* p_src, const size_t w, const size_t h)
{
	bool bret = true;
	size_t size_arr = w * h;
	size_t size_mem = size_arr * sizeof(K);
	K* p_tmp_ll = new K[size_arr];
	memset(p_tmp_ll, 0, size_mem);

	//Transform of HaarWavelet
	wavelet_haar_lv1<T, K>(p_tmp_ll, p_dst_lh, p_dst_hl, p_dst_hh, p_src, w, h);
	wavelet_haar_lv1<K, K>(p_dst_ll2, p_dst_lh2, p_dst_hl2, p_dst_hh2, p_tmp_ll, w, h);


	delete p_tmp_ll;
	return bret;
}