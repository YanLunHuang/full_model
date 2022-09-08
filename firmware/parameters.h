#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_deepcalo.h"
#include "nnet_utils/nnet_deepcalo_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_image.h"
#include "nnet_utils/nnet_image_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"
#include "nnet_utils/nnet_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w117.h"
#include "weights/b117.h"
#include "weights/w118.h"
#include "weights/b118.h"
#include "weights/w119.h"
#include "weights/b119.h"
#include "weights/w14.h"
#include "weights/b14.h"
#include "weights/w16.h"
#include "weights/b16.h"
#include "weights/s19.h"
#include "weights/b19.h"
#include "weights/w20.h"
#include "weights/b20.h"
//#include "weights/w25.h"
#include "weights/b25.h"
#include "weights/s27.h"
#include "weights/b27.h"
//#include "weights/w30.h"
#include "weights/b30.h"
#include "weights/s33.h"
#include "weights/b33.h"
#include "weights/s34.h"
#include "weights/b34.h"
#include "weights/w36.h"
#include "weights/b36.h"
//#include "weights/w38.h"
#include "weights/b38.h"
#include "weights/s40.h"
#include "weights/b40.h"
#include "weights/w45.h"
#include "weights/b45.h"
#include "weights/s48.h"
#include "weights/b48.h"
#include "weights/w52.h"
#include "weights/b52.h"
#include "weights/s54.h"
#include "weights/b54.h"
#include "weights/w57.h"
#include "weights/b57.h"
#include "weights/s60.h"
#include "weights/b60.h"
#include "weights/w64.h"
#include "weights/b64.h"
#include "weights/s66.h"
#include "weights/b66.h"
//#include "weights/w69.h"
#include "weights/b69.h"
#include "weights/s72.h"
#include "weights/b72.h"
//#include "weights/w76.h"
#include "weights/b76.h"
#include "weights/s78.h"
#include "weights/b78.h"
//#include "weights/w81.h"
#include "weights/b81.h"
#include "weights/s83.h"
#include "weights/b83.h"
//#include "weights/w87.h"
#include "weights/b87.h"
#include "weights/s89.h"
#include "weights/b89.h"
//#include "weights/w92.h"
#include "weights/b92.h"
#include "weights/s94.h"
#include "weights/b94.h"
//#include "weights/w96.h"
#include "weights/b96.h"
#include "weights/s98.h"
#include "weights/b98.h"
#include "weights/w100.h"
#include "weights/b100.h"

//hls-fpga-machine-learning insert layer-config
// conv1d_1
struct config117_mult : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 768;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias117_t bias_t;
    typedef weight117_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config117 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 57;
    static const unsigned n_chan = 6;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 57;
    static const unsigned reuse_factor = 768;
    static const unsigned n_zeros = 138;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 1;
    static const ap_uint<filt_width> pixels[min_width];
    typedef model_default_t accum_t;
    typedef bias117_t bias_t;
    typedef weight117_t weight_t;
    typedef config117_mult mult_config;
};
const ap_uint<config117::filt_width> config117::pixels[] = {1};

// leaky_re_lu_1
struct LeakyReLU_config4 : nnet::activ_config {
    static const unsigned n_in = 7296;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_1_table_t table_t;
};

// mask_track
struct config5 : nnet::mask_track_config {	 
    static const unsigned n_in = N_INPUT_1_1*N_INPUT_2_1;
    static const unsigned height = 57;
    static const unsigned width = 6;
    static const unsigned n_chan = 6;
};

// conv1d_2
struct config118_mult : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 2048;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias118_t bias_t;
    typedef weight118_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config118 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 57;
    static const unsigned n_chan = 128;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 57;
    static const unsigned reuse_factor = 2048;
    static const unsigned n_zeros = 2274;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 1;
    static const ap_uint<filt_width> pixels[min_width];
    typedef model_default_t accum_t;
    typedef bias118_t bias_t;
    typedef weight118_t weight_t;
    typedef config118_mult mult_config;
};
const ap_uint<config118::filt_width> config118::pixels[] = {1};

// conv1d
struct config119_mult : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 128;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias119_t bias_t;
    typedef weight119_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config119 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 57;
    static const unsigned n_chan = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 57;
    static const unsigned reuse_factor = 128;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 1;
    static const ap_uint<filt_width> pixels[min_width];
    typedef model_default_t accum_t;
    typedef bias119_t bias_t;
    typedef weight119_t weight_t;
    typedef config119_mult mult_config;
};
const ap_uint<config119::filt_width> config119::pixels[] = {1};

// leaky_re_lu_2
struct LeakyReLU_config10 : nnet::activ_config {
    static const unsigned n_in = 7296;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_2_table_t table_t;
};

// multiply
struct config11 : nnet::merge_config {
    static const unsigned n_elem = N_OUTPUTS_119*N_FILT_119;
};

// sum1d
struct config12 : nnet::sum1d_config {
    static const unsigned n_in = N_OUTPUTS_8*N_FILT_8;
    static const unsigned height = 57;
    static const unsigned width = 128;
    static const unsigned n_chan = 128;
    static const unsigned n_filt = 128;
};

// dense_1
struct config14 : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 128;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 128;
    static const unsigned n_zeros = 1143;
    static const unsigned n_nonzeros = 15241;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias14_t bias_t;
    typedef weight14_t weight_t;
    typedef layer14_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense
struct config16 : nnet::dense_config {
    static const unsigned n_in = 15;
    static const unsigned n_out = 256;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 325;
    static const unsigned n_nonzeros = 3515;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias16_t bias_t;
    typedef weight16_t weight_t;
    typedef layer16_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// leaky_re_lu_3
struct LeakyReLU_config18 : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_3_table_t table_t;
};

// batch_normalization
struct config19 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_16;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_2
struct config20 : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 128;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 128;
    static const unsigned n_zeros = 1145;
    static const unsigned n_nonzeros = 15239;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias20_t bias_t;
    typedef weight20_t weight_t;
    typedef layer20_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// leaky_re_lu
struct LeakyReLU_config22 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_table_t table_t;
};

// leaky_re_lu_4
struct LeakyReLU_config23 : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_4_table_t table_t;
};

// concatenate
struct config24 : nnet::concat_config {
    static const unsigned n_elem1_0 = 256;
    static const unsigned n_elem1_1 = 0;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 128;
    static const unsigned n_elem2_1 = 0;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// dense_3
struct config25 : nnet::dense_config {
    static const unsigned n_in = 384;
    static const unsigned n_out = 512;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 512;
    static const unsigned n_zeros = 24341;
    static const unsigned n_nonzeros = 172267;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias25_t bias_t;
    typedef weight25_t weight_t;
    typedef layer25_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// batch_normalization_2
struct config27 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_25;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// leaky_re_lu_5
struct LeakyReLU_config28 : nnet::activ_config {
    static const unsigned n_in = 512;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_5_table_t table_t;
};

// dense_4
struct config30 : nnet::dense_config {
    static const unsigned n_in = 512;
    static const unsigned n_out = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1024;
    static const unsigned n_zeros = 95783;
    static const unsigned n_nonzeros = 428505;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias30_t bias_t;
    typedef weight30_t weight_t;
    typedef layer30_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// up_sampling2d
struct config32 : nnet::resize_config {
    static const unsigned height = 56;
    static const unsigned width = 11;
    static const unsigned n_chan = 4;
    static const unsigned new_height = 56;
    static const unsigned new_width = 55;
};

// batch_normalization_3
struct config33 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_30;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// batch_normalization_1
struct config34 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_32*OUT_WIDTH_32*N_CHAN_32;
    static const unsigned n_filt = 4;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// zp2d_conv2d
struct config108 : nnet::padding2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 55;
    static const unsigned n_chan = 4;
    static const unsigned out_height = 60;
    static const unsigned out_width = 59;
    static const unsigned pad_top = 2;
    static const unsigned pad_bottom = 2;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 2;
};

// leaky_re_lu_6
struct LeakyReLU_config35 : nnet::activ_config {
    static const unsigned n_in = 1024;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_6_table_t table_t;
};

// conv2d
struct config36_mult : nnet::dense_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 100;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias36_t bias_t;
    typedef weight36_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config36 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 60;
    static const unsigned in_width = 59;
    static const unsigned n_chan = 4;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 56;
    static const unsigned out_width = 55;
    static const unsigned reuse_factor = 100;
    static const unsigned n_zeros = 193;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 9;
    static const unsigned min_width = 9;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias36_t bias_t;
    typedef weight36_t weight_t;
    typedef config36_mult mult_config;
};
const ap_uint<config36::filt_height * config36::filt_width> config36::pixels[] = {1,3,7,15,31,30,28,24,16,33,99,231,495,1023,990,924,792,528,1057,3171,7399,15855,32767,31710,29596,25368,16912,33825,101475,236775,507375,1048575,1014750,947100,811800,541200,1082401,3247203,7576807,16236015,33554431,32472030,30307228,25977624,17318416,1082400,3247200,7576800,16236000,33554400,32472000,30307200,25977600,17318400,1082368,3247104,7576576,16235520,33553408,32471040,30306304,25976832,17317888,1081344,3244032,7569408,16220160,33521664,32440320,30277632,25952256,17301504,1048576,3145728,7340032,15728640,32505856,31457280,29360128,25165824,16777216};

// dense_5
struct config38 : nnet::dense_config {
    static const unsigned n_in = 1024;
    static const unsigned n_out = 992;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 992;
    static const unsigned n_zeros = 145980;
    static const unsigned n_nonzeros = 869828;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias38_t bias_t;
    typedef weight38_t weight_t;
    typedef layer38_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// batch_normalization_4
struct config40 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_36*OUT_WIDTH_36*N_FILT_36;
    static const unsigned n_filt = 16;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// slice_tensor1d
struct config41 : nnet::slice_tensor1d_config {	 
    static const unsigned n_in = N_LAYER_38;
    static const unsigned start = 0;
    static const unsigned end = 32;	
};

// fi_lm
struct config42 : nnet::film_config {
    static const unsigned height = 56;
    static const unsigned width = 55;
    static const unsigned n_chan1 = 16;
    static const unsigned n_chan2 = 32;
    static const unsigned n_inp1 = OUT_HEIGHT_36*OUT_WIDTH_36*N_FILT_36;
    static const unsigned n_inp2 = N_OUTPUTS_1_41;
};

// leaky_re_lu_7
struct LeakyReLU_config43 : nnet::activ_config {
    static const unsigned n_in = 49280;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_7_table_t table_t;
};

// max_pooling2d
struct config44 : nnet::pooling2d_config {
    static const unsigned in_height = 56;
    static const unsigned in_width = 55;
    static const unsigned n_filt = 16;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 28;
    static const unsigned out_width = 27;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 2000;
    typedef model_default_t accum_t;
};

// zp2d_conv2d_1
struct config109 : nnet::padding2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 27;
    static const unsigned n_chan = 16;
    static const unsigned out_height = 30;
    static const unsigned out_width = 29;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_1
struct config45_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 144;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias45_t bias_t;
    typedef weight45_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config45 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 30;
    static const unsigned in_width = 29;
    static const unsigned n_chan = 16;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 28;
    static const unsigned out_width = 27;
    static const unsigned reuse_factor = 144;
    static const unsigned n_zeros = 666;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias45_t bias_t;
    typedef weight45_t weight_t;
    typedef config45_mult mult_config;
};
const ap_uint<config45::filt_height * config45::filt_width> config45::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// slice_tensor1d_1
struct config47 : nnet::slice_tensor1d_config {	 
    static const unsigned n_in = N_LAYER_38;
    static const unsigned start = 32;
    static const unsigned end = 992;	
};

// batch_normalization_5
struct config48 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_45*OUT_WIDTH_45*N_FILT_45;
    static const unsigned n_filt = 32;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// slice_tensor1d_2
struct config49 : nnet::slice_tensor1d_config {	 
    static const unsigned n_in = N_OUTPUTS_1_47;
    static const unsigned start = 0;
    static const unsigned end = 64;	
};

// fi_lm_1
struct config50 : nnet::film_config {
    static const unsigned height = 28;
    static const unsigned width = 27;
    static const unsigned n_chan1 = 32;
    static const unsigned n_chan2 = 64;
    static const unsigned n_inp1 = OUT_HEIGHT_45*OUT_WIDTH_45*N_FILT_45;
    static const unsigned n_inp2 = N_OUTPUTS_1_49;
};

// leaky_re_lu_8
struct LeakyReLU_config51 : nnet::activ_config {
    static const unsigned n_in = 24192;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_8_table_t table_t;
};

// zp2d_conv2d_2
struct config110 : nnet::padding2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 27;
    static const unsigned n_chan = 32;
    static const unsigned out_height = 30;
    static const unsigned out_width = 29;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_2
struct config52_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 288;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias52_t bias_t;
    typedef weight52_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config52 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 30;
    static const unsigned in_width = 29;
    static const unsigned n_chan = 32;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 28;
    static const unsigned out_width = 27;
    static const unsigned reuse_factor = 288;
    static const unsigned n_zeros = 1868;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias52_t bias_t;
    typedef weight52_t weight_t;
    typedef config52_mult mult_config;
};
const ap_uint<config52::filt_height * config52::filt_width> config52::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// batch_normalization_6
struct config54 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_52*OUT_WIDTH_52*N_FILT_52;
    static const unsigned n_filt = 32;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// leaky_re_lu_9
struct LeakyReLU_config55 : nnet::activ_config {
    static const unsigned n_in = 24192;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_9_table_t table_t;
};

// max_pooling2d_1
struct config56 : nnet::pooling2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 27;
    static const unsigned n_filt = 32;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 14;
    static const unsigned out_width = 13;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 2000;
    typedef model_default_t accum_t;
};

// zp2d_conv2d_3
struct config111 : nnet::padding2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 13;
    static const unsigned n_chan = 32;
    static const unsigned out_height = 16;
    static const unsigned out_width = 15;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_3
struct config57_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 288;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias57_t bias_t;
    typedef weight57_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config57 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 16;
    static const unsigned in_width = 15;
    static const unsigned n_chan = 32;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 14;
    static const unsigned out_width = 13;
    static const unsigned reuse_factor = 288;
    static const unsigned n_zeros = 3818;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias57_t bias_t;
    typedef weight57_t weight_t;
    typedef config57_mult mult_config;
};
const ap_uint<config57::filt_height * config57::filt_width> config57::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// slice_tensor1d_3
struct config59 : nnet::slice_tensor1d_config {	 
    static const unsigned n_in = N_OUTPUTS_1_47;
    static const unsigned start = 64;
    static const unsigned end = 960;	
};

// batch_normalization_7
struct config60 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57;
    static const unsigned n_filt = 64;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// slice_tensor1d_4
struct config61 : nnet::slice_tensor1d_config {	 
    static const unsigned n_in = N_OUTPUTS_1_59;
    static const unsigned start = 0;
    static const unsigned end = 128;	
};

// fi_lm_2
struct config62 : nnet::film_config {
    static const unsigned height = 14;
    static const unsigned width = 13;
    static const unsigned n_chan1 = 64;
    static const unsigned n_chan2 = 128;
    static const unsigned n_inp1 = OUT_HEIGHT_57*OUT_WIDTH_57*N_FILT_57;
    static const unsigned n_inp2 = N_OUTPUTS_1_61;
};

// leaky_re_lu_10
struct LeakyReLU_config63 : nnet::activ_config {
    static const unsigned n_in = 11648;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_10_table_t table_t;
};

// zp2d_conv2d_4
struct config112 : nnet::padding2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 13;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 16;
    static const unsigned out_width = 15;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_4
struct config64_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 576;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias64_t bias_t;
    typedef weight64_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config64 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 16;
    static const unsigned in_width = 15;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 14;
    static const unsigned out_width = 13;
    static const unsigned reuse_factor = 576;
    static const unsigned n_zeros = 10249;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias64_t bias_t;
    typedef weight64_t weight_t;
    typedef config64_mult mult_config;
};
const ap_uint<config64::filt_height * config64::filt_width> config64::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// batch_normalization_8
struct config66 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_64*OUT_WIDTH_64*N_FILT_64;
    static const unsigned n_filt = 64;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// leaky_re_lu_11
struct LeakyReLU_config67 : nnet::activ_config {
    static const unsigned n_in = 11648;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_11_table_t table_t;
};

// max_pooling2d_2
struct config68 : nnet::pooling2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 13;
    static const unsigned n_filt = 64;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 7;
    static const unsigned out_width = 6;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 2000;
    typedef model_default_t accum_t;
};

// zp2d_conv2d_5
struct config113 : nnet::padding2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 6;
    static const unsigned n_chan = 64;
    static const unsigned out_height = 9;
    static const unsigned out_width = 8;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_5
struct config69_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 576;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias69_t bias_t;
    typedef weight69_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config69 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 8;
    static const unsigned n_chan = 64;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 6;
    static const unsigned reuse_factor = 576;
    static const unsigned n_zeros = 20705;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias69_t bias_t;
    typedef weight69_t weight_t;
    typedef config69_mult mult_config;
};
const ap_uint<config69::filt_height * config69::filt_width> config69::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// slice_tensor1d_5
struct config71 : nnet::slice_tensor1d_config {	 
    static const unsigned n_in = N_OUTPUTS_1_59;
    static const unsigned start = 128;
    static const unsigned end = 896;	
};

// batch_normalization_9
struct config72 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_69*OUT_WIDTH_69*N_FILT_69;
    static const unsigned n_filt = 128;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// slice_tensor1d_6
struct config73 : nnet::slice_tensor1d_config {	 
    static const unsigned n_in = N_OUTPUTS_1_71;
    static const unsigned start = 0;
    static const unsigned end = 256;	
};

// fi_lm_3
struct config74 : nnet::film_config {
    static const unsigned height = 7;
    static const unsigned width = 6;
    static const unsigned n_chan1 = 128;
    static const unsigned n_chan2 = 256;
    static const unsigned n_inp1 = OUT_HEIGHT_69*OUT_WIDTH_69*N_FILT_69;
    static const unsigned n_inp2 = N_OUTPUTS_1_73;
};

// leaky_re_lu_12
struct LeakyReLU_config75 : nnet::activ_config {
    static const unsigned n_in = 5376;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_12_table_t table_t;
};

// zp2d_conv2d_6
struct config114 : nnet::padding2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 6;
    static const unsigned n_chan = 128;
    static const unsigned out_height = 9;
    static const unsigned out_width = 8;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_6
struct config76_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 1152;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias76_t bias_t;
    typedef weight76_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config76 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 9;
    static const unsigned in_width = 8;
    static const unsigned n_chan = 128;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 7;
    static const unsigned out_width = 6;
    static const unsigned reuse_factor = 1152;
    static const unsigned n_zeros = 55977;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias76_t bias_t;
    typedef weight76_t weight_t;
    typedef config76_mult mult_config;
};
const ap_uint<config76::filt_height * config76::filt_width> config76::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// batch_normalization_10
struct config78 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_76*OUT_WIDTH_76*N_FILT_76;
    static const unsigned n_filt = 128;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// leaky_re_lu_13
struct LeakyReLU_config79 : nnet::activ_config {
    static const unsigned n_in = 5376;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_13_table_t table_t;
};

// max_pooling2d_3
struct config80 : nnet::pooling2d_config {
    static const unsigned in_height = 7;
    static const unsigned in_width = 6;
    static const unsigned n_filt = 128;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = pool_height;
    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned out_height = 3;
    static const unsigned out_width = 3;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 2000;
    typedef model_default_t accum_t;
};

// zp2d_conv2d_7
struct config115 : nnet::padding2d_config {
    static const unsigned in_height = 3;
    static const unsigned in_width = 3;
    static const unsigned n_chan = 128;
    static const unsigned out_height = 5;
    static const unsigned out_width = 5;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_7
struct config81_mult : nnet::dense_config {
    static const unsigned n_in = 1152;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 1152;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias81_t bias_t;
    typedef weight81_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config81 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 5;
    static const unsigned in_width = 5;
    static const unsigned n_chan = 128;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 3;
    static const unsigned out_width = 3;
    static const unsigned reuse_factor = 1152;
    static const unsigned n_zeros = 113169;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias81_t bias_t;
    typedef weight81_t weight_t;
    typedef config81_mult mult_config;
};
const ap_uint<config81::filt_height * config81::filt_width> config81::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// batch_normalization_11
struct config83 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_81*OUT_WIDTH_81*N_FILT_81;
    static const unsigned n_filt = 256;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// slice_tensor1d_7
struct config84 : nnet::slice_tensor1d_config {	 
    static const unsigned n_in = N_OUTPUTS_1_71;
    static const unsigned start = 256;
    static const unsigned end = 768;	
};

// fi_lm_4
struct config85 : nnet::film_config {
    static const unsigned height = 3;
    static const unsigned width = 3;
    static const unsigned n_chan1 = 256;
    static const unsigned n_chan2 = 512;
    static const unsigned n_inp1 = OUT_HEIGHT_81*OUT_WIDTH_81*N_FILT_81;
    static const unsigned n_inp2 = N_OUTPUTS_1_84;
};

// leaky_re_lu_14
struct LeakyReLU_config86 : nnet::activ_config {
    static const unsigned n_in = 2304;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_14_table_t table_t;
};

// zp2d_conv2d_8
struct config116 : nnet::padding2d_config {
    static const unsigned in_height = 3;
    static const unsigned in_width = 3;
    static const unsigned n_chan = 256;
    static const unsigned out_height = 5;
    static const unsigned out_width = 5;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_8
struct config87_mult : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 2304;
    static const unsigned strategy = nnet::resource;
    typedef model_default_t accum_t;
    typedef bias87_t bias_t;
    typedef weight87_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config87 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = 5;
    static const unsigned in_width = 5;
    static const unsigned n_chan = 256;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = 3;
    static const unsigned out_width = 3;
    static const unsigned reuse_factor = 2304;
    static const unsigned n_zeros = 299836;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef model_default_t accum_t;
    typedef bias87_t bias_t;
    typedef weight87_t weight_t;
    typedef config87_mult mult_config;
};
const ap_uint<config87::filt_height * config87::filt_width> config87::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// batch_normalization_12
struct config89 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_87*OUT_WIDTH_87*N_FILT_87;
    static const unsigned n_filt = 256;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// leaky_re_lu_15
struct LeakyReLU_config90 : nnet::activ_config {
    static const unsigned n_in = 2304;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_15_table_t table_t;
};

// dense_6
struct config92 : nnet::dense_config {
    static const unsigned n_in = 2304;
    static const unsigned n_out = 256;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 159213;
    static const unsigned n_nonzeros = 430611;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias92_t bias_t;
    typedef weight92_t weight_t;
    typedef layer92_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// batch_normalization_13
struct config94 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_92;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// leaky_re_lu_16
struct LeakyReLU_config95 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_16_table_t table_t;
};

// dense_7
struct config96 : nnet::dense_config {
    static const unsigned n_in = 256;
    static const unsigned n_out = 256;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 256;
    static const unsigned n_zeros = 6313;
    static const unsigned n_nonzeros = 59223;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias96_t bias_t;
    typedef weight96_t weight_t;
    typedef layer96_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// batch_normalization_14
struct config98 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_96;
    static const unsigned n_filt = -1;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// leaky_re_lu_17
struct LeakyReLU_config99 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef leaky_re_lu_17_table_t table_t;
};

// dense_8
struct config100 : nnet::dense_config {
    static const unsigned n_in = 256;
    static const unsigned n_out = 1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias100_t bias_t;
    typedef weight100_t weight_t;
    typedef layer100_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// activation
struct relu_config102 : nnet::activ_config {
    static const unsigned n_in = 1;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 2000;
    typedef activation_table_t table_t;
};


#endif
