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
    hls::stream<input1_T> data1[1],
    hls::stream<input2_T> data2[1],
    hls::stream<res_T> res[1])
{
	
	input2_T scalar_gamma;
	res_T out_data;
	
	input2_T in_data2[CONFIG_T::n_chan2];
	for (int i = 0; i < CONFIG_T::n_chan2; i++) {
		#pragma HLS PIPELINE II=1
		in_data2[i] = data2[0].read();
	}
	
	for (int i = 0; i < CONFIG_T::n_inp1/CONFIG_T::n_chan1; i++) {
		
		FilmPack: for (int k = 0; k < CONFIG_T::n_chan1; k++) {
			#pragma HLS PIPELINE II=1
			input1_T in_data1 = data1[0].read();
			out_data = in_data1*(in_data2[k]+1) + in_data2[CONFIG_T::n_chan1+k];
			res[0].write(out_data);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//for switch
template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void film_single(
    hls::stream<input1_T> data1[1],
    hls::stream<input2_T> data2[CONFIG_T::data_transfer_in2],
    hls::stream<res_T> res[1])
{
    //unuse
    input2_T scalar_gamma;

    input2_T in_data2[CONFIG_T::n_chan2];
    if(CONFIG_T::data_transfer_in2 != 1) {
        #pragma HLS ARRAY_PARTITION variable=in_data2 complete
    }

    for (int i = 0; i < CONFIG_T::n_chan2; i++) {
        if(CONFIG_T::data_transfer_in2 == 1){
            #pragma HLS PIPELINE II=1
            in_data2[i] = data2[0].read();
        }else {
            #pragma HLS UNROLL
            in_data2[i] = data2[i].read();
        }
    }

    for (int i = 0; i < CONFIG_T::n_inp1/CONFIG_T::n_chan1; i++) {
        for (int k = 0; k < CONFIG_T::n_chan1; k++) {
            #pragma HLS PIPELINE II=1
            input1_T in_data1 = data1[0].read();
            res_T out_data = in_data1*(in_data2[k]+1) + in_data2[CONFIG_T::n_chan1+k];
            res[0].write(out_data);
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void film_array(
    hls::stream<input1_T> data1[CONFIG_T::data_transfer_in1],
    hls::stream<input2_T> data2[CONFIG_T::data_transfer_in2],
    hls::stream<res_T> res[CONFIG_T::data_transfer_out])
{
    //unuse
    input2_T scalar_gamma;

    input2_T in_data2[CONFIG_T::n_chan2];
    #pragma HLS ARRAY_RESHAPE variable=in_data2 complete

    for (int i = 0; i < CONFIG_T::n_chan2; i++) {
        if(CONFIG_T::data_transfer_in2 == 1){
            #pragma HLS PIPELINE II=1
            in_data2[i] = data2[0].read();
        }else {
            #pragma HLS UNROLL
            in_data2[i] = data2[i].read();
        }
    }

    for (int i = 0; i < CONFIG_T::n_inp1/CONFIG_T::n_chan1; i++) {
        #pragma HLS PIPELINE II=1
        for (int k = 0; k < CONFIG_T::n_chan1; k++) {
            #pragma HLS UNROLL
            input1_T in_data1 = data1[k].read();
            res_T out_data = in_data1*(in_data2[k]+1) + in_data2[CONFIG_T::n_chan1+k];
            res[k].write(out_data);
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void film_switch(
    hls::stream<input1_T> data1[CONFIG_T::data_transfer_in1],
    hls::stream<input2_T> data2[CONFIG_T::data_transfer_in2],
    hls::stream<res_T> res[CONFIG_T::data_transfer_out])
{
    #pragma HLS inline region
    if(CONFIG_T::data_transfer_out == 1){
        film_single<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }else {
        film_array<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
        hls::stream<data_T> data[1],
        hls::stream<res_T> res[1]) {
        
        res_T out_data;
        MaskLoop1: for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {
            #pragma HLS PIPELINE II=CONFIG_T::n_chan
            MaskLoop2: for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
                data_T in_data = data[0].read();
                if(in_data==0){
                    out_data = 0;
                }else{
                    out_data = 1;
                }
            }
            res[0].write(out_data);
        }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//for switch
template<class data_T, class res_T, typename CONFIG_T>
void mask_track_single(
        hls::stream<data_T> data[1],
        hls::stream<res_T> res[1]) {
        
        res_T out_data;
        MaskLoop1: for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {
            MaskLoop2: for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
                #pragma HLS PIPELINE II=1
                data_T in_data = data[0].read();
                if(in_data==0){
                    out_data = 0;
                }else{
                    out_data = 1;
                }
            }
            res[0].write(out_data);
        }
}

template<class data_T, class res_T, typename CONFIG_T>
void mask_track_array(
        hls::stream<data_T> data[CONFIG_T::n_chan],
        hls::stream<res_T> res[1]) {
        
        data_T in_data[CONFIG_T::n_chan];
        #pragma HLS ARRAY_PARTITION variable=in_data complete
        res_T out_data;

        MaskLoop1: for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {
            #pragma HLS PIPELINE II=1

            MaskLoop2: for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
                #pragma HLS UNROLL
                in_data[k] = data[k].read();
            }

            MaskLoop3: for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
                #pragma HLS UNROLL
                if(in_data[k]==0){
                    out_data = 0;
                }else{
                    out_data = 1;
                }
            }
            res[0].write(out_data);
        }
}

template<class data_T, class res_T, typename CONFIG_T>
void mask_track_switch(
    hls::stream<data_T> data[CONFIG_T::data_transfer_out],
    hls::stream<res_T> res[1]) {

    #pragma HLS inline region
    if(CONFIG_T::data_transfer_out == 1){
        mask_track_single<data_T, res_T, CONFIG_T>(data, res);
    }else {
        mask_track_array<data_T, res_T, CONFIG_T>(data, res);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<class data_T, class res_T, typename CONFIG_T>
void sum1d(
        hls::stream<data_T> &data,
        hls::stream<res_T> &res) 
{

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
        hls::stream<data_T> data[1],
        hls::stream<res_T> res[1]) 
{
    res_T out_data[CONFIG_T::n_filt];

    for (unsigned i = 0; i < CONFIG_T::n_filt; i++) out_data[i] = 0;

    for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {

        for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
            #pragma HLS PIPELINE II=1
            data_T in_data = data[0].read();
                if (CONFIG_T::n_filt == 1) 
                    out_data[0] = out_data[0] + in_data;
                else                  
                    out_data[k] = out_data[k] + in_data;
        }
    }
    for (unsigned i = 0; i < CONFIG_T::n_filt; i++){
        #pragma HLS PIPELINE II=1
        res[0].write(out_data[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//for switch
template<class data_T, class res_T, typename CONFIG_T>
void sum1d_s2s(
    hls::stream<data_T> data[1],
    hls::stream<res_T> res[1]) 
{
    res_T out_data[CONFIG_T::n_filt];

    for (unsigned i = 0; i < CONFIG_T::n_filt; i++) out_data[i] = 0;

    for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {

        for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
            #pragma HLS PIPELINE II=1
            data_T in_data = data[0].read();
            if (CONFIG_T::n_filt == 1) 
                out_data[0] = out_data[0] + in_data;
            else                  
                out_data[k] = out_data[k] + in_data;
        }
    }
    for (unsigned i = 0; i < CONFIG_T::n_filt; i++){
        #pragma HLS PIPELINE II=1
        res[0].write(out_data[i]);
    }
}
/*
template<class data_T, class res_T, typename CONFIG_T>
void sum1d_s2a(
        hls::stream<data_T> data[1],
        hls::stream<res_T> res[CONFIG_T::n_filt]) {

    data_T in_data[CONFIG_T::n_chan];
    res_T out_data[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=in_data complete
    #pragma HLS ARRAY_PARTITION variable=out_data complete

    for (unsigned i = 0; i < CONFIG_T::n_filt; i++) out_data[i] = 0;

    for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {

        for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
            #pragma HLS PIPELINE II=1
            data_T in_data = data[0].read();
                if (CONFIG_T::n_filt == 1) 
                    out_data[0] = out_data[0] + in_data;
                else                  
                    out_data[k] = out_data[k] + in_data;
        }
    }
    for (unsigned i = 0; i < CONFIG_T::n_filt; i++){
        #pragma HLS UNROLL
        res[i].write(out_data[i]);
    }
}
*/

template<class data_T, class res_T, typename CONFIG_T>
void sum1d_a2a(
    hls::stream<data_T> data[CONFIG_T::n_chan],
    hls::stream<res_T> res[CONFIG_T::n_filt]) 
{
    res_T out_data[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=out_data complete

    for (unsigned i = 0; i < CONFIG_T::n_filt; i++) out_data[i] = 0;

    for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {
        #pragma HLS PIPELINE
        for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
            #pragma HLS UNROLL
            data_T in_data = data[k].read();
            if (CONFIG_T::n_filt == 1) 
                out_data[0] = out_data[0] + in_data;
            else                  
                out_data[k] = out_data[k] + in_data;
        }
    }
    for (unsigned i = 0; i < CONFIG_T::n_filt; i++){
        #pragma HLS UNROLL
        res[i].write(out_data[i]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void sum1d_a2s(
    hls::stream<data_T> data[CONFIG_T::n_chan],
    hls::stream<res_T> res[1]) 
{
    res_T out_data[CONFIG_T::n_filt];
	#pragma HLS ARRAY_RESHAPE variable=out_data complete

    for (unsigned i = 0; i < CONFIG_T::n_filt; i++) out_data[i] = 0;

    for (unsigned i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {
        #pragma HLS PIPELINE
        for (unsigned k = 0; k < CONFIG_T::n_chan; k++){
            #pragma HLS UNROLL
            data_T in_data = data[k].read();
            if (CONFIG_T::n_filt == 1) 
                out_data[0] = out_data[0] + in_data;
            else                  
                out_data[k] = out_data[k] + in_data;
        }
    }
    for (unsigned i = 0; i < CONFIG_T::n_filt; i++){
        #pragma HLS PIPELINE II=1
        res[0].write(out_data[i]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void sum1d_switch(
    hls::stream<data_T> data[CONFIG_T::data_transfer_in],
    hls::stream<res_T> res[CONFIG_T::data_transfer_out]) 
{
    #pragma HLS inline region
    if(CONFIG_T::data_transfer_in == 1 && CONFIG_T::data_transfer_out == 1){
        sum1d_s2s<data_T, res_T, CONFIG_T>(data, res);
    }/*else if(CONFIG_T::data_transfer_in == 1 && CONFIG_T::data_transfer_out != 1){
        sum1d_s2a<data_T, res_T, CONFIG_T>(data, res);
    }*/else if(CONFIG_T::data_transfer_in != 1 && CONFIG_T::data_transfer_out != 1){
        sum1d_a2a<data_T, res_T, CONFIG_T>(data, res);
    }else {
        sum1d_a2s<data_T, res_T, CONFIG_T>(data, res);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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
        hls::stream<data_T> data[1],
        hls::stream<res_T> res[1]) {
        

	for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS PIPELINE II=1
		data_T in_data = data[0].read();
		if ( i >=  CONFIG_T::start && i < CONFIG_T::end ){
			res_T out_data = in_data;
			res[0].write(out_data);
		}
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//for switch
template<class data_T, class res_T, typename CONFIG_T>
void slice_tensor1d_s2s(
    hls::stream<data_T> data[1],
    hls::stream<res_T> res[1]) 
{
   for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS PIPELINE II=1
        data_T in_data = data[0].read();
        if ( i >=  CONFIG_T::start && i < CONFIG_T::end ){
           res_T out_data = in_data;
           res[0].write(out_data);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void slice_tensor1d_s2a(
    hls::stream<data_T> data[1],
    hls::stream<res_T> res[CONFIG_T::n_filt]) 
{
    int count = 0;
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS PIPELINE II=1
        data_T in_data = data[0].read();
        if ( i >=  CONFIG_T::start && i < CONFIG_T::end ){
            res_T out_data = in_data;
            res[count++].write(out_data);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void slice_tensor1d_a2a(
    hls::stream<data_T> data[CONFIG_T::n_in],
    hls::stream<res_T> res[CONFIG_T::n_filt]) 
{
    int count = 0;
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        data_T in_data = data[i].read();
        if ( i >=  CONFIG_T::start && i < CONFIG_T::end ){
            res_T out_data = in_data;
            res[count++].write(out_data);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void slice_tensor1d_switch(
    hls::stream<data_T> data[CONFIG_T::data_transfer_in],
    hls::stream<res_T> res[CONFIG_T::data_transfer_out]) 
{
    #pragma HLS inline region
    if(CONFIG_T::data_transfer_in == 1 && CONFIG_T::data_transfer_out == 1){
        slice_tensor1d_s2s<data_T, res_T, CONFIG_T>(data, res);
    }else if(CONFIG_T::data_transfer_in == 1 && CONFIG_T::data_transfer_out != 1){
        slice_tensor1d_s2a<data_T, res_T, CONFIG_T>(data, res);
    }else if(CONFIG_T::data_transfer_in != 1 && CONFIG_T::data_transfer_out != 1){
        slice_tensor1d_a2a<data_T, res_T, CONFIG_T>(data, res);
    }else {
        assert( (CONFIG_T::data_transfer_in != 1 && CONFIG_T::data_transfer_out == 1) 
             && "data_transfer_in & data_transfer_out have to be adjusted correctly");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


}
#endif
