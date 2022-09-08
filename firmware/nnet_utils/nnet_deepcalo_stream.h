#ifndef NNET_DEEPCALO_STREAM_H_
#define NNET_DEEPCALO_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <cmath>
namespace nnet {


template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void film(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
	
	input2_T scalar_gamma;
	res_T out_data;
	
	input2_T in_data2 = data2.read();	
	for (int i = 0; i < CONFIG_T::n_inp1/input1_T::size; i++) {
		
		input1_T in_data1 = data1.read();
		
		FilmPack: for (int k = 0; k < input1_T::size; k++) {
			
			
			out_data[k] = in_data1[k]*(in_data2[k]+1) + in_data2[input1_T::size+k];
		}
		
		res.write(out_data);
		
	}			
					
}

template<class input1_T, size_t SIZE1, class input2_T, size_t SIZE2, class res_T, typename CONFIG_T>
void film_s(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
	
	input2_T scalar_gamma;
	res_T out_data;
	
	input2_T in_data2[SIZE2];
	for (int i = 0; i < SIZE2; i++) {
		in_data2[i] = data2.read();
	}
	
	for (int i = 0; i < CONFIG_T::n_inp1/SIZE1; i++) {
		
		FilmPack: for (int k = 0; k < SIZE1; k++) {
			
			input1_T in_data1 = data1.read();
			out_data[k] = in_data1*(in_data2[k]+1) + in_data2[SIZE1+k];
		}
		
		res.write(out_data);
		
	}			
					
}


template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void film_ss(
    hls::stream<input1_T> &data1,
    hls::stream<input2_T> &data2,
    hls::stream<res_T> &res)
{
	
	input2_T scalar_gamma;
	res_T out_data;
	
	input2_T in_data2[CONFIG_T::n_chan2];
	for (int i = 0; i < CONFIG_T::n_chan2; i++) {
		#pragma HLS PIPELINE II=1
		in_data2[i] = data2.read();
	}
	
	for (int i = 0; i < CONFIG_T::n_inp1/CONFIG_T::n_chan1; i++) {
		
		FilmPack: for (int k = 0; k < CONFIG_T::n_chan1; k++) {
			#pragma HLS PIPELINE II=1
			input1_T in_data1 = data1.read();
			out_data = in_data1*(in_data2[k]+1) + in_data2[CONFIG_T::n_chan1+k];
			res.write(out_data);
		}
	}
}


template<class data_T, class res_T, typename CONFIG_T>
void mask_track(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {
        #pragma HLS PIPELINE
 


    	MaskLoop1: for (unsigned i = 0; i < CONFIG_T::n_in/data_T::size; i++) {
        	#pragma HLS UNROLL
        	data_T in_data = data.read();   
			
        	res_T out_data;
			#pragma HLS DATA_PACK variable=out_data

        	MaskLoop2: for (unsigned k = 0; k < data_T::size; k++){
        	
               	#pragma HLS UNROLL
				if(in_data[k]==0){
					out_data[0] = 0;
				}else{
					out_data[0] = 1;
				}
            }
            res.write(out_data);
    	}
}


template<class data_T, class res_T, typename CONFIG_T>
void mask_track_ss(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {
        
        res_T out_data;
        MaskLoop1: for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {
            #pragma HLS PIPELINE II=CONFIG_T::n_chan
            MaskLoop2: for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
                data_T in_data = data.read();
                if(in_data==0){
                    out_data = 0;
                }else{
                    out_data = 1;
                }
            }
            res.write(out_data);
        }
}

template<class data_T, class res_T, typename CONFIG_T>
void sum1d(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {


	res_T out_data;
	#pragma HLS DATA_PACK variable=out_data

	for (unsigned i = 0; i < res_T::size; i++)	out_data[i] = 0;

    	for (unsigned i = 0; i < CONFIG_T::n_in/data_T::size; i++) {
        	#pragma HLS UNROLL
        	data_T in_data = data.read();

        	for (unsigned k = 0; k < data_T::size; k++){
            		if (res_T::size == 1) 
            			out_data[0] = out_data[0] + in_data[k];
            		else                  
            			out_data[k] = out_data[k] + in_data[k];
       	 }
	}
    	res.write(out_data);
}


template<class data_T, class res_T, typename CONFIG_T>
void sum1d_ss(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {

    res_T out_data[CONFIG_T::n_filt];

    for (unsigned i = 0; i < CONFIG_T::n_filt; i++) out_data[i] = 0;

    for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {

        for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
            #pragma HLS PIPELINE II=1
            data_T in_data = data.read();
                if (CONFIG_T::n_filt == 1) 
                    out_data[0] = out_data[0] + in_data;
                else                  
                    out_data[k] = out_data[k] + in_data;
        }
    }
    for (unsigned i = 0; i < CONFIG_T::n_filt; i++){
        #pragma HLS PIPELINE II=1
        res.write(out_data[i]);
    }
}



template<class data_T, class res_T, typename CONFIG_T>
void slice_tensor1d(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {
        
	data_T in_data = data.read();
	res_T out_data;
	#pragma HLS DATA_PACK variable=out_data

	for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
		if ( i >=  CONFIG_T::start && i < CONFIG_T::end ){
			out_data[i-CONFIG_T::start] = in_data[i];
		}
    }
	res.write(out_data);
}

template<class data_T, class res_T, typename CONFIG_T>
void slice_tensor1d_s(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {
        
	data_T in_data[CONFIG_T::n_in];
	for(int i = 0;i < CONFIG_T::n_in;i++){
		in_data[i] = data.read();
	}
	res_T out_data;
	#pragma HLS DATA_PACK variable=out_data

	for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
		if ( i >=  CONFIG_T::start && i < CONFIG_T::end ){
			out_data[i-CONFIG_T::start] = in_data[i];
		}
    }
	res.write(out_data);
}


template<class data_T, class res_T, typename CONFIG_T>
void slice_tensor1d_ss(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) {
        

	for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS PIPELINE II=1
		data_T in_data = data.read();
		if ( i >=  CONFIG_T::start && i < CONFIG_T::end ){
			res_T out_data = in_data;
			res.write(out_data);
		}
    }
}


}
#endif
