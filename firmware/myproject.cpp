//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input29_t> &em_barrel, hls::stream<input13_t> &scalars, hls::stream<input_t> &tracks,
    hls::stream<result_t> &layer102_out,
	weight25_t w25[196608],
	weight30_t w30[524288],
	weight38_t w38[1015808],
	weight69_t w69[73728],
	weight76_t w76[147456],
	weight81_t w81[294912],
	weight87_t w87[589824],
	weight92_t w92[589824],
	weight96_t w96[65536]
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=em_barrel,scalars,tracks,layer102_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight117_t, 768>(w117, "w117.txt");
        nnet::load_weights_from_txt<bias117_t, 128>(b117, "b117.txt");
        nnet::load_weights_from_txt<weight118_t, 16384>(w118, "w118.txt");
        nnet::load_weights_from_txt<bias118_t, 128>(b118, "b118.txt");
        nnet::load_weights_from_txt<weight119_t, 128>(w119, "w119.txt");
        nnet::load_weights_from_txt<bias119_t, 128>(b119, "b119.txt");
        nnet::load_weights_from_txt<weight14_t, 16384>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 128>(b14, "b14.txt");
        nnet::load_weights_from_txt<weight16_t, 3840>(w16, "w16.txt");
        nnet::load_weights_from_txt<bias16_t, 256>(b16, "b16.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s19, "s19.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b19, "b19.txt");
        nnet::load_weights_from_txt<weight20_t, 16384>(w20, "w20.txt");
        nnet::load_weights_from_txt<bias20_t, 128>(b20, "b20.txt");
        //nnet::load_weights_from_txt<weight25_t, 196608>(w25, "w25.txt");
        nnet::load_weights_from_txt<bias25_t, 512>(b25, "b25.txt");
        nnet::load_weights_from_txt<model_default_t, 512>(s27, "s27.txt");
        nnet::load_weights_from_txt<model_default_t, 512>(b27, "b27.txt");
        //nnet::load_weights_from_txt<weight30_t, 524288>(w30, "w30.txt");
        nnet::load_weights_from_txt<bias30_t, 1024>(b30, "b30.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(s33, "s33.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(b33, "b33.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(s34, "s34.txt");
        nnet::load_weights_from_txt<model_default_t, 4>(b34, "b34.txt");
        nnet::load_weights_from_txt<weight36_t, 1600>(w36, "w36.txt");
        nnet::load_weights_from_txt<bias36_t, 16>(b36, "b36.txt");
        //nnet::load_weights_from_txt<weight38_t, 1015808>(w38, "w38.txt");
        nnet::load_weights_from_txt<bias38_t, 992>(b38, "b38.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(s40, "s40.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b40, "b40.txt");
        nnet::load_weights_from_txt<weight45_t, 4608>(w45, "w45.txt");
        nnet::load_weights_from_txt<bias45_t, 32>(b45, "b45.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(s48, "s48.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b48, "b48.txt");
        nnet::load_weights_from_txt<weight52_t, 9216>(w52, "w52.txt");
        nnet::load_weights_from_txt<bias52_t, 32>(b52, "b52.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(s54, "s54.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b54, "b54.txt");
        nnet::load_weights_from_txt<weight57_t, 18432>(w57, "w57.txt");
        nnet::load_weights_from_txt<bias57_t, 64>(b57, "b57.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s60, "s60.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b60, "b60.txt");
        nnet::load_weights_from_txt<weight64_t, 36864>(w64, "w64.txt");
        nnet::load_weights_from_txt<bias64_t, 64>(b64, "b64.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s66, "s66.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b66, "b66.txt");
        //nnet::load_weights_from_txt<weight69_t, 73728>(w69, "w69.txt");
        nnet::load_weights_from_txt<bias69_t, 128>(b69, "b69.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s72, "s72.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b72, "b72.txt");
        //nnet::load_weights_from_txt<weight76_t, 147456>(w76, "w76.txt");
        nnet::load_weights_from_txt<bias76_t, 128>(b76, "b76.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(s78, "s78.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b78, "b78.txt");
        //nnet::load_weights_from_txt<weight81_t, 294912>(w81, "w81.txt");
        nnet::load_weights_from_txt<bias81_t, 256>(b81, "b81.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s83, "s83.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b83, "b83.txt");
        //nnet::load_weights_from_txt<weight87_t, 589824>(w87, "w87.txt");
        nnet::load_weights_from_txt<bias87_t, 256>(b87, "b87.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s89, "s89.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b89, "b89.txt");
        //nnet::load_weights_from_txt<weight92_t, 589824>(w92, "w92.txt");
        nnet::load_weights_from_txt<bias92_t, 256>(b92, "b92.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s94, "s94.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b94, "b94.txt");
        //nnet::load_weights_from_txt<weight96_t, 65536>(w96, "w96.txt");
        nnet::load_weights_from_txt<bias96_t, 256>(b96, "b96.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(s98, "s98.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(b98, "b98.txt");
        nnet::load_weights_from_txt<weight100_t, 256>(w100, "w100.txt");
        nnet::load_weights_from_txt<bias100_t, 1>(b100, "b100.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer103_t> layer103_cpy1("layer103_cpy1");
    #pragma HLS STREAM variable=layer103_cpy1 depth=57
    hls::stream<layer103_t> layer103_cpy2("layer103_cpy2");
    #pragma HLS STREAM variable=layer103_cpy2 depth=57
    nnet::clone_stream_ss<input_t, layer103_t, 342>(tracks, layer103_cpy1, layer103_cpy2); // clone_tracks

    hls::stream<layer117_t> layer117_out("layer117_out");
    #pragma HLS STREAM variable=layer117_out depth=57
    nnet::pointwise_conv_1d_cl_ss<layer103_t, layer117_t, config117>(layer103_cpy1, layer117_out, w117, b117); // conv1d_1

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=57
    nnet::leaky_relu_ss<layer117_t, layer4_t, LeakyReLU_config4>(layer117_out, 0.30000001192092896, layer4_out); // leaky_re_lu_1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=57
    nnet::mask_track_ss<layer103_t, layer5_t, config5>(layer103_cpy2, layer5_out); // mask_track

    hls::stream<layer118_t> layer118_out("layer118_out");
    #pragma HLS STREAM variable=layer118_out depth=57
    nnet::pointwise_conv_1d_cl_ss<layer4_t, layer118_t, config118>(layer4_out, layer118_out, w118, b118); // conv1d_2

    hls::stream<layer119_t> layer119_out("layer119_out");
    #pragma HLS STREAM variable=layer119_out depth=57
    nnet::pointwise_conv_1d_cl_ss<layer5_t, layer119_t, config119>(layer5_out, layer119_out, w119, b119); // conv1d

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=57
    nnet::leaky_relu_ss<layer118_t, layer10_t, LeakyReLU_config10>(layer118_out, 0.30000001192092896, layer10_out); // leaky_re_lu_2

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=57
    nnet::multiply_ss<layer119_t, layer10_t, layer11_t, config11>(layer119_out, layer10_out, layer11_out); // multiply

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::sum1d_ss<layer11_t, layer12_t, config12>(layer11_out, layer12_out); // sum1d

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=1
    nnet::dense_ss<layer12_t, layer14_t, config14>(layer12_out, layer14_out, w14, b14); // dense_1

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=1
    nnet::dense_ss<input13_t, layer16_t, config16>(scalars, layer16_out, w16, b16); // dense

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=1
    nnet::leaky_relu_ss<layer14_t, layer18_t, LeakyReLU_config18>(layer14_out, 0.30000001192092896, layer18_out); // leaky_re_lu_3

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=1
    nnet::normalize_ss<layer16_t, layer19_t, config19>(layer16_out, layer19_out, s19, b19); // batch_normalization

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=1
    nnet::dense_ss<layer18_t, layer20_t, config20>(layer18_out, layer20_out, w20, b20); // dense_2

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=1
    nnet::leaky_relu_ss<layer19_t, layer22_t, LeakyReLU_config22>(layer19_out, 0.30000001192092896, layer22_out); // leaky_re_lu

    hls::stream<layer23_t> layer23_out("layer23_out");
    #pragma HLS STREAM variable=layer23_out depth=1
    nnet::leaky_relu_ss<layer20_t, layer23_t, LeakyReLU_config23>(layer20_out, 0.30000001192092896, layer23_out); // leaky_re_lu_4

    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=1
    nnet::concatenate1d_ss<layer22_t, layer23_t, layer24_t, config24>(layer22_out, layer23_out, layer24_out); // concatenate

    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=1
    nnet::dense_ss<layer24_t, layer25_t, config25>(layer24_out, layer25_out, w25, b25); // dense_3

    hls::stream<layer27_t> layer27_out("layer27_out");
    #pragma HLS STREAM variable=layer27_out depth=1
    nnet::normalize_ss<layer25_t, layer27_t, config27>(layer25_out, layer27_out, s27, b27); // batch_normalization_2

    hls::stream<layer28_t> layer28_out("layer28_out");
    #pragma HLS STREAM variable=layer28_out depth=1
    nnet::leaky_relu_ss<layer27_t, layer28_t, LeakyReLU_config28>(layer27_out, 0.30000001192092896, layer28_out); // leaky_re_lu_5

    hls::stream<layer30_t> layer30_out("layer30_out");
    #pragma HLS STREAM variable=layer30_out depth=1
    nnet::dense_ss<layer28_t, layer30_t, config30>(layer28_out, layer30_out, w30, b30); // dense_4

    hls::stream<layer32_t> layer32_out("layer32_out");
    #pragma HLS STREAM variable=layer32_out depth=3080
    nnet::resize_nearest_ss<input29_t, config32>(em_barrel, layer32_out); // up_sampling2d

    hls::stream<layer33_t> layer33_out("layer33_out");
    #pragma HLS STREAM variable=layer33_out depth=1
    nnet::normalize_ss<layer30_t, layer33_t, config33>(layer30_out, layer33_out, s33, b33); // batch_normalization_3

    hls::stream<layer34_t> layer34_out("layer34_out");
    #pragma HLS STREAM variable=layer34_out depth=3080
    nnet::normalize_ss<layer32_t, layer34_t, config34>(layer32_out, layer34_out, s34, b34); // batch_normalization_1

    hls::stream<layer108_t> layer108_out("layer108_out");
    #pragma HLS STREAM variable=layer108_out depth=3540
    nnet::zeropad2d_cl_ss<layer34_t, layer108_t, config108>(layer34_out, layer108_out); // zp2d_conv2d

    hls::stream<layer35_t> layer35_out("layer35_out");
    #pragma HLS STREAM variable=layer35_out depth=1
    nnet::leaky_relu_ss<layer33_t, layer35_t, LeakyReLU_config35>(layer33_out, 0.30000001192092896, layer35_out); // leaky_re_lu_6

    hls::stream<layer36_t> layer36_out("layer36_out");
    #pragma HLS STREAM variable=layer36_out depth=3080
    nnet::conv_2d_cl_ss<layer108_t, layer36_t, config36>(layer108_out, layer36_out, w36, b36); // conv2d

    hls::stream<layer38_t> layer38_out("layer38_out");
    #pragma HLS STREAM variable=layer38_out depth=1
    nnet::dense_ss<layer35_t, layer38_t, config38>(layer35_out, layer38_out, w38, b38); // dense_5

    hls::stream<layer104_t> layer104_cpy1("layer104_cpy1");
    #pragma HLS STREAM variable=layer104_cpy1 depth=1
    hls::stream<layer104_t> layer104_cpy2("layer104_cpy2");
    #pragma HLS STREAM variable=layer104_cpy2 depth=1
    nnet::clone_stream_ss<layer38_t, layer104_t, 992>(layer38_out, layer104_cpy1, layer104_cpy2); // clone_dense_5

    hls::stream<layer40_t> layer40_out("layer40_out");
    #pragma HLS STREAM variable=layer40_out depth=3080
    nnet::normalize_ss<layer36_t, layer40_t, config40>(layer36_out, layer40_out, s40, b40); // batch_normalization_4

    hls::stream<layer41_t> layer41_out("layer41_out");
    #pragma HLS STREAM variable=layer41_out depth=1
    nnet::slice_tensor1d_ss<layer104_t, layer41_t, config41>(layer104_cpy1, layer41_out); // slice_tensor1d

    hls::stream<layer42_t> layer42_out("layer42_out");
    #pragma HLS STREAM variable=layer42_out depth=3080
    nnet::film_ss<layer40_t, layer41_t, layer42_t, config42>(layer40_out, layer41_out, layer42_out); // fi_lm

    hls::stream<layer43_t> layer43_out("layer43_out");
    #pragma HLS STREAM variable=layer43_out depth=3080
    nnet::leaky_relu_ss<layer42_t, layer43_t, LeakyReLU_config43>(layer42_out, 0.30000001192092896, layer43_out); // leaky_re_lu_7

    hls::stream<layer44_t> layer44_out("layer44_out");
    #pragma HLS STREAM variable=layer44_out depth=756
    nnet::pooling_ss<layer43_t, layer44_t, config44>(layer43_out, layer44_out); // max_pooling2d

    hls::stream<layer109_t> layer109_out("layer109_out");
    #pragma HLS STREAM variable=layer109_out depth=870
    nnet::zeropad2d_cl_ss<layer44_t, layer109_t, config109>(layer44_out, layer109_out); // zp2d_conv2d_1

    hls::stream<layer45_t> layer45_out("layer45_out");
    #pragma HLS STREAM variable=layer45_out depth=756
    nnet::conv_2d_cl_ss<layer109_t, layer45_t, config45>(layer109_out, layer45_out, w45, b45); // conv2d_1

    hls::stream<layer47_t> layer47_out("layer47_out");
    #pragma HLS STREAM variable=layer47_out depth=1
    nnet::slice_tensor1d_ss<layer104_t, layer47_t, config47>(layer104_cpy2, layer47_out); // slice_tensor1d_1

    hls::stream<layer105_t> layer105_cpy1("layer105_cpy1");
    #pragma HLS STREAM variable=layer105_cpy1 depth=1
    hls::stream<layer105_t> layer105_cpy2("layer105_cpy2");
    #pragma HLS STREAM variable=layer105_cpy2 depth=1
    nnet::clone_stream_ss<layer47_t, layer105_t, 960>(layer47_out, layer105_cpy1, layer105_cpy2); // clone_slice_tensor1d_1

    hls::stream<layer48_t> layer48_out("layer48_out");
    #pragma HLS STREAM variable=layer48_out depth=756
    nnet::normalize_ss<layer45_t, layer48_t, config48>(layer45_out, layer48_out, s48, b48); // batch_normalization_5

    hls::stream<layer49_t> layer49_out("layer49_out");
    #pragma HLS STREAM variable=layer49_out depth=1
    nnet::slice_tensor1d_ss<layer105_t, layer49_t, config49>(layer105_cpy1, layer49_out); // slice_tensor1d_2

    hls::stream<layer50_t> layer50_out("layer50_out");
    #pragma HLS STREAM variable=layer50_out depth=756
    nnet::film_ss<layer48_t, layer49_t, layer50_t, config50>(layer48_out, layer49_out, layer50_out); // fi_lm_1

    hls::stream<layer51_t> layer51_out("layer51_out");
    #pragma HLS STREAM variable=layer51_out depth=756
    nnet::leaky_relu_ss<layer50_t, layer51_t, LeakyReLU_config51>(layer50_out, 0.30000001192092896, layer51_out); // leaky_re_lu_8

    hls::stream<layer110_t> layer110_out("layer110_out");
    #pragma HLS STREAM variable=layer110_out depth=870
    nnet::zeropad2d_cl_ss<layer51_t, layer110_t, config110>(layer51_out, layer110_out); // zp2d_conv2d_2

    hls::stream<layer52_t> layer52_out("layer52_out");
    #pragma HLS STREAM variable=layer52_out depth=756
    nnet::conv_2d_cl_ss<layer110_t, layer52_t, config52>(layer110_out, layer52_out, w52, b52); // conv2d_2

    hls::stream<layer54_t> layer54_out("layer54_out");
    #pragma HLS STREAM variable=layer54_out depth=756
    nnet::normalize_ss<layer52_t, layer54_t, config54>(layer52_out, layer54_out, s54, b54); // batch_normalization_6

    hls::stream<layer55_t> layer55_out("layer55_out");
    #pragma HLS STREAM variable=layer55_out depth=756
    nnet::leaky_relu_ss<layer54_t, layer55_t, LeakyReLU_config55>(layer54_out, 0.30000001192092896, layer55_out); // leaky_re_lu_9

    hls::stream<layer56_t> layer56_out("layer56_out");
    #pragma HLS STREAM variable=layer56_out depth=182
    nnet::pooling_ss<layer55_t, layer56_t, config56>(layer55_out, layer56_out); // max_pooling2d_1

    hls::stream<layer111_t> layer111_out("layer111_out");
    #pragma HLS STREAM variable=layer111_out depth=240
    nnet::zeropad2d_cl_ss<layer56_t, layer111_t, config111>(layer56_out, layer111_out); // zp2d_conv2d_3

    hls::stream<layer57_t> layer57_out("layer57_out");
    #pragma HLS STREAM variable=layer57_out depth=182
    nnet::conv_2d_cl_ss<layer111_t, layer57_t, config57>(layer111_out, layer57_out, w57, b57); // conv2d_3

    hls::stream<layer59_t> layer59_out("layer59_out");
    #pragma HLS STREAM variable=layer59_out depth=1
    nnet::slice_tensor1d_ss<layer105_t, layer59_t, config59>(layer105_cpy2, layer59_out); // slice_tensor1d_3

    hls::stream<layer106_t> layer106_cpy1("layer106_cpy1");
    #pragma HLS STREAM variable=layer106_cpy1 depth=1
    hls::stream<layer106_t> layer106_cpy2("layer106_cpy2");
    #pragma HLS STREAM variable=layer106_cpy2 depth=1
    nnet::clone_stream_ss<layer59_t, layer106_t, 896>(layer59_out, layer106_cpy1, layer106_cpy2); // clone_slice_tensor1d_3

    hls::stream<layer60_t> layer60_out("layer60_out");
    #pragma HLS STREAM variable=layer60_out depth=182
    nnet::normalize_ss<layer57_t, layer60_t, config60>(layer57_out, layer60_out, s60, b60); // batch_normalization_7

    hls::stream<layer61_t> layer61_out("layer61_out");
    #pragma HLS STREAM variable=layer61_out depth=1
    nnet::slice_tensor1d_ss<layer106_t, layer61_t, config61>(layer106_cpy1, layer61_out); // slice_tensor1d_4

    hls::stream<layer62_t> layer62_out("layer62_out");
    #pragma HLS STREAM variable=layer62_out depth=182
    nnet::film_ss<layer60_t, layer61_t, layer62_t, config62>(layer60_out, layer61_out, layer62_out); // fi_lm_2

    hls::stream<layer63_t> layer63_out("layer63_out");
    #pragma HLS STREAM variable=layer63_out depth=182
    nnet::leaky_relu_ss<layer62_t, layer63_t, LeakyReLU_config63>(layer62_out, 0.30000001192092896, layer63_out); // leaky_re_lu_10

    hls::stream<layer112_t> layer112_out("layer112_out");
    #pragma HLS STREAM variable=layer112_out depth=240
    nnet::zeropad2d_cl_ss<layer63_t, layer112_t, config112>(layer63_out, layer112_out); // zp2d_conv2d_4

    hls::stream<layer64_t> layer64_out("layer64_out");
    #pragma HLS STREAM variable=layer64_out depth=182
    nnet::conv_2d_cl_ss<layer112_t, layer64_t, config64>(layer112_out, layer64_out, w64, b64); // conv2d_4

    hls::stream<layer66_t> layer66_out("layer66_out");
    #pragma HLS STREAM variable=layer66_out depth=182
    nnet::normalize_ss<layer64_t, layer66_t, config66>(layer64_out, layer66_out, s66, b66); // batch_normalization_8

    hls::stream<layer67_t> layer67_out("layer67_out");
    #pragma HLS STREAM variable=layer67_out depth=182
    nnet::leaky_relu_ss<layer66_t, layer67_t, LeakyReLU_config67>(layer66_out, 0.30000001192092896, layer67_out); // leaky_re_lu_11

    hls::stream<layer68_t> layer68_out("layer68_out");
    #pragma HLS STREAM variable=layer68_out depth=42
    nnet::pooling_ss<layer67_t, layer68_t, config68>(layer67_out, layer68_out); // max_pooling2d_2

    hls::stream<layer113_t> layer113_out("layer113_out");
    #pragma HLS STREAM variable=layer113_out depth=72
    nnet::zeropad2d_cl_ss<layer68_t, layer113_t, config113>(layer68_out, layer113_out); // zp2d_conv2d_5

    hls::stream<layer69_t> layer69_out("layer69_out");
    #pragma HLS STREAM variable=layer69_out depth=42
    nnet::conv_2d_cl_ss<layer113_t, layer69_t, config69>(layer113_out, layer69_out, w69, b69); // conv2d_5

    hls::stream<layer71_t> layer71_out("layer71_out");
    #pragma HLS STREAM variable=layer71_out depth=1
    nnet::slice_tensor1d_ss<layer106_t, layer71_t, config71>(layer106_cpy2, layer71_out); // slice_tensor1d_5

    hls::stream<layer107_t> layer107_cpy1("layer107_cpy1");
    #pragma HLS STREAM variable=layer107_cpy1 depth=1
    hls::stream<layer107_t> layer107_cpy2("layer107_cpy2");
    #pragma HLS STREAM variable=layer107_cpy2 depth=1
    nnet::clone_stream_ss<layer71_t, layer107_t, 768>(layer71_out, layer107_cpy1, layer107_cpy2); // clone_slice_tensor1d_5

    hls::stream<layer72_t> layer72_out("layer72_out");
    #pragma HLS STREAM variable=layer72_out depth=42
    nnet::normalize_ss<layer69_t, layer72_t, config72>(layer69_out, layer72_out, s72, b72); // batch_normalization_9

    hls::stream<layer73_t> layer73_out("layer73_out");
    #pragma HLS STREAM variable=layer73_out depth=1
    nnet::slice_tensor1d_ss<layer107_t, layer73_t, config73>(layer107_cpy1, layer73_out); // slice_tensor1d_6

    hls::stream<layer74_t> layer74_out("layer74_out");
    #pragma HLS STREAM variable=layer74_out depth=42
    nnet::film_ss<layer72_t, layer73_t, layer74_t, config74>(layer72_out, layer73_out, layer74_out); // fi_lm_3

    hls::stream<layer75_t> layer75_out("layer75_out");
    #pragma HLS STREAM variable=layer75_out depth=42
    nnet::leaky_relu_ss<layer74_t, layer75_t, LeakyReLU_config75>(layer74_out, 0.30000001192092896, layer75_out); // leaky_re_lu_12

    hls::stream<layer114_t> layer114_out("layer114_out");
    #pragma HLS STREAM variable=layer114_out depth=72
    nnet::zeropad2d_cl_ss<layer75_t, layer114_t, config114>(layer75_out, layer114_out); // zp2d_conv2d_6

    hls::stream<layer76_t> layer76_out("layer76_out");
    #pragma HLS STREAM variable=layer76_out depth=42
    nnet::conv_2d_cl_ss<layer114_t, layer76_t, config76>(layer114_out, layer76_out, w76, b76); // conv2d_6

    hls::stream<layer78_t> layer78_out("layer78_out");
    #pragma HLS STREAM variable=layer78_out depth=42
    nnet::normalize_ss<layer76_t, layer78_t, config78>(layer76_out, layer78_out, s78, b78); // batch_normalization_10

    hls::stream<layer79_t> layer79_out("layer79_out");
    #pragma HLS STREAM variable=layer79_out depth=42
    nnet::leaky_relu_ss<layer78_t, layer79_t, LeakyReLU_config79>(layer78_out, 0.30000001192092896, layer79_out); // leaky_re_lu_13

    hls::stream<layer80_t> layer80_out("layer80_out");
    #pragma HLS STREAM variable=layer80_out depth=9
    nnet::pooling_ss<layer79_t, layer80_t, config80>(layer79_out, layer80_out); // max_pooling2d_3

    hls::stream<layer115_t> layer115_out("layer115_out");
    #pragma HLS STREAM variable=layer115_out depth=25
    nnet::zeropad2d_cl_ss<layer80_t, layer115_t, config115>(layer80_out, layer115_out); // zp2d_conv2d_7

    hls::stream<layer81_t> layer81_out("layer81_out");
    #pragma HLS STREAM variable=layer81_out depth=9
    nnet::conv_2d_cl_ss<layer115_t, layer81_t, config81>(layer115_out, layer81_out, w81, b81); // conv2d_7

    hls::stream<layer83_t> layer83_out("layer83_out");
    #pragma HLS STREAM variable=layer83_out depth=9
    nnet::normalize_ss<layer81_t, layer83_t, config83>(layer81_out, layer83_out, s83, b83); // batch_normalization_11

    hls::stream<layer84_t> layer84_out("layer84_out");
    #pragma HLS STREAM variable=layer84_out depth=1
    nnet::slice_tensor1d_ss<layer107_t, layer84_t, config84>(layer107_cpy2, layer84_out); // slice_tensor1d_7

    hls::stream<layer85_t> layer85_out("layer85_out");
    #pragma HLS STREAM variable=layer85_out depth=9
    nnet::film_ss<layer83_t, layer84_t, layer85_t, config85>(layer83_out, layer84_out, layer85_out); // fi_lm_4

    hls::stream<layer86_t> layer86_out("layer86_out");
    #pragma HLS STREAM variable=layer86_out depth=9
    nnet::leaky_relu_ss<layer85_t, layer86_t, LeakyReLU_config86>(layer85_out, 0.30000001192092896, layer86_out); // leaky_re_lu_14

    hls::stream<layer116_t> layer116_out("layer116_out");
    #pragma HLS STREAM variable=layer116_out depth=25
    nnet::zeropad2d_cl_ss<layer86_t, layer116_t, config116>(layer86_out, layer116_out); // zp2d_conv2d_8

    hls::stream<layer87_t> layer87_out("layer87_out");
    #pragma HLS STREAM variable=layer87_out depth=9
    nnet::conv_2d_cl_ss<layer116_t, layer87_t, config87>(layer116_out, layer87_out, w87, b87); // conv2d_8

    hls::stream<layer89_t> layer89_out("layer89_out");
    #pragma HLS STREAM variable=layer89_out depth=9
    nnet::normalize_ss<layer87_t, layer89_t, config89>(layer87_out, layer89_out, s89, b89); // batch_normalization_12

    hls::stream<layer90_t> layer90_out("layer90_out");
    #pragma HLS STREAM variable=layer90_out depth=9
    nnet::leaky_relu_ss<layer89_t, layer90_t, LeakyReLU_config90>(layer89_out, 0.30000001192092896, layer90_out); // leaky_re_lu_15

    hls::stream<layer92_t> layer92_out("layer92_out");
    #pragma HLS STREAM variable=layer92_out depth=1
    nnet::dense_ss<layer90_t, layer92_t, config92>(layer90_out, layer92_out, w92, b92); // dense_6

    hls::stream<layer94_t> layer94_out("layer94_out");
    #pragma HLS STREAM variable=layer94_out depth=1
    nnet::normalize_ss<layer92_t, layer94_t, config94>(layer92_out, layer94_out, s94, b94); // batch_normalization_13

    hls::stream<layer95_t> layer95_out("layer95_out");
    #pragma HLS STREAM variable=layer95_out depth=1
    nnet::leaky_relu_ss<layer94_t, layer95_t, LeakyReLU_config95>(layer94_out, 0.30000001192092896, layer95_out); // leaky_re_lu_16

    hls::stream<layer96_t> layer96_out("layer96_out");
    #pragma HLS STREAM variable=layer96_out depth=1
    nnet::dense_ss<layer95_t, layer96_t, config96>(layer95_out, layer96_out, w96, b96); // dense_7

    hls::stream<layer98_t> layer98_out("layer98_out");
    #pragma HLS STREAM variable=layer98_out depth=1
    nnet::normalize_ss<layer96_t, layer98_t, config98>(layer96_out, layer98_out, s98, b98); // batch_normalization_14

    hls::stream<layer99_t> layer99_out("layer99_out");
    #pragma HLS STREAM variable=layer99_out depth=1
    nnet::leaky_relu_ss<layer98_t, layer99_t, LeakyReLU_config99>(layer98_out, 0.30000001192092896, layer99_out); // leaky_re_lu_17

    hls::stream<layer100_t> layer100_out("layer100_out");
    #pragma HLS STREAM variable=layer100_out depth=1
    nnet::dense_ss<layer99_t, layer100_t, config100>(layer99_out, layer100_out, w100, b100); // dense_8

    nnet::relu_ss<layer100_t, result_t, relu_config102>(layer100_out, layer102_out); // activation

}
