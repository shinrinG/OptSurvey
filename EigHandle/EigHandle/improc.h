#pragma once
#include "defs.h"

template<class T,class K>
bool wavelet_haar_core_horizon(K* p_dst_p,K*p_dst_n ,const T* p_src, const size_t w, const size_t h);

template<class T, class K>
bool wavelet_haar_core_vertical(K* p_dst_p, K* p_dst_n, const T* p_src, const size_t w, const size_t h);

template<class T, class K>
bool wavelet_haar_lv1(K* p_dst_ll,K* p_dst_lh,K* p_dst_hl, K* p_dst_hh, const T* p_src, const size_t w, const size_t h);

template<class T, class K>
bool wavelet_haar_lv2(K* p_dst_ll2, K* p_dst_lh2, K* p_dst_hl2, K* p_dst_hh2, K* p_dst_lh, K* p_dst_hl, K* p_dst_hh, const T* p_src, const size_t w, const size_t h);