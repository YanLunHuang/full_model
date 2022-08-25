#ifndef NNET_IMAGE_STREAM_H_
#define NNET_IMAGE_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void resize_nearest(
    hls::stream<data_T> &image,
    hls::stream<data_T> &resized
) {
	assert(CONFIG_T::new_height % CONFIG_T::height == 0);
	assert(CONFIG_T::new_width % CONFIG_T::width == 0);
	constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
	constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;
	
	
	ImageHeight: for (unsigned h = 0; h < CONFIG_T::height; h++) {
		#pragma HLS PIPELINE
	
		data_T data_in_row[CONFIG_T::width];
		
		ImageWidth: for (unsigned i = 0; i < CONFIG_T::width; i++) {
			#pragma HLS UNROLL
			
			data_T  in_data = image.read();
			
			ImageChan: for (unsigned j = 0; j < CONFIG_T::n_chan; j++) {
				#pragma HLS UNROLL
				
				data_in_row[i][j] = in_data[j];
			}
		}
		
		ResizeHeight: for (unsigned i = 0; i <ratio_height; i++) {
			#pragma HLS UNROLL
			
			ImageWidth2: for (unsigned l = 0; l < CONFIG_T::width; l++) {
				#pragma HLS UNROLL
				
				ResizeWidth: for (unsigned j = 0; j < ratio_width; j++) {
					#pragma HLS UNROLL
				
					data_T out_data;
					#pragma HLS DATA_PACK variable=out_data
				
					ResizeChan: for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
						#pragma HLS UNROLL
					
						out_data[k] = data_in_row[l][k];
					}
					
					resized.write(out_data);   
				}
			}
		}
	}
}


template<class data_T, typename CONFIG_T>
void resize_nearest_ss(
    hls::stream<data_T> &image,
    hls::stream<data_T> &resized
) {
	assert(CONFIG_T::new_height % CONFIG_T::height == 0);
	assert(CONFIG_T::new_width % CONFIG_T::width == 0);
	constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
	constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;
	constexpr unsigned ii = ratio_height * ratio_width;
	
	int current_chan = 0;int current_width = 0;
	int current_chan2 = 0;int current_width2 = 0;
	
	data_T data_in_row[CONFIG_T::width][CONFIG_T::n_chan];
	
	ImageHeight: for (unsigned h = 0; h < CONFIG_T::height; h++) {
		#pragma HLS PIPELINE II=ii
		
		ReadData: for(unsigned i = 0; i < CONFIG_T::n_chan*CONFIG_T::width ; i++){
			#pragma HLS UNROLL
			current_chan = i % CONFIG_T::n_chan;
			current_width = i / CONFIG_T::n_chan;
			data_in_row[current_width][current_chan] = image.read();
		}
		
		RatioHeight: for (unsigned i = 0; i <ratio_height; i++) {
			#pragma HLS UNROLL
			ImageWidth: for (unsigned k = 0; k < CONFIG_T::n_chan*ratio_width*CONFIG_T::width; k++) {
				#pragma HLS UNROLL
				current_chan2 = k % CONFIG_T::n_chan;
				current_width2 = k / (CONFIG_T::n_chan*ratio_width);
				data_T out_data = data_in_row[current_width2][current_chan2];
				resized.write(out_data);   
			}
		}
	}
}



}

#endif
