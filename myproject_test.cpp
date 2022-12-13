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
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"

//hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

namespace nnet {
    bool trace_enabled = true;
    std::map<std::string, void *> *trace_outputs = NULL;
    size_t trace_type_size = sizeof(double);
}

template<class T, size_t SIZE>
void load_weights_from_txt(T *w, const char* fname) {

    std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname) << " does not exist" << std::endl;
        exit(1);
    }

    std::string line;
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token;

        size_t i = 0;
        while(std::getline(iss, token, ',')) {
            std::istringstream(token) >> w[i];
            i++;
        }

        if (SIZE != i) {
            std::cerr << "ERROR: Expected " << SIZE << " values";
            std::cerr << " but read only " << i << " values" << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
  //load input data from text file
  std::ifstream fin("tb_data/tb_input_features.dat");
  //load predictions from text file
  std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
  std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
  std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
  std::ofstream fout(RESULTS_LOG);

  std::string iline;
  std::string pline;
  int e = 0;

  if (fin.is_open() && fpr.is_open()) {
    while ( std::getline(fin,iline) && std::getline (fpr,pline) ) {
      if (e % CHECKPOINT == 0) std::cout << "Processing input " << e << std::endl;
      char* cstr=const_cast<char*>(iline.c_str());
      char* current;
      std::vector<float> in;
      current=strtok(cstr," ");
      while(current!=NULL) {
        in.push_back(atof(current));
        current=strtok(NULL," ");
      }
      cstr=const_cast<char*>(pline.c_str());
      std::vector<float> pr;
      current=strtok(cstr," ");
      while(current!=NULL) {
        pr.push_back(atof(current));
        current=strtok(NULL," ");
      }

      //hls-fpga-machine-learning insert data
      hls::stream<input29_t> em_barrel[4];
      nnet::copy_data_switch<float, input29_t, 0, N_INPUT_1_29*N_INPUT_2_29*N_INPUT_3_29, 4>(in, em_barrel);
      hls::stream<input13_t> scalars[1];
      nnet::copy_data_s<float, input13_t, 2464, N_INPUT_1_13>(in, scalars);
      hls::stream<input_t> tracks[6];
      nnet::copy_data_switch<float, input_t, 2479, N_INPUT_1_1*N_INPUT_2_1, 6>(in, tracks);
      hls::stream<result_t> layer102_out[1];
      
      //load weight file
      weight25_t w25[196608];
      weight30_t w30[524288];
      weight38_t w38[1015808];
      weight69_t w69[73728];
      weight76_t w76[147456];
      weight81_t w81[294912];
      weight87_t w87[589824];
      weight92_t w92[589824];
      weight96_t w96[65536];
      nnet::load_weights_from_txt<weight25_t, 196608>(w25, "w25.txt");
      nnet::load_weights_from_txt<weight30_t, 524288>(w30, "w30.txt");
      nnet::load_weights_from_txt<weight38_t, 1015808>(w38, "w38.txt");
      nnet::load_weights_from_txt<weight69_t, 73728>(w69, "w69.txt");
      nnet::load_weights_from_txt<weight76_t, 147456>(w76, "w76.txt");
      nnet::load_weights_from_txt<weight81_t, 294912>(w81, "w81.txt");
      nnet::load_weights_from_txt<weight87_t, 589824>(w87, "w87.txt");
      nnet::load_weights_from_txt<weight92_t, 589824>(w92, "w92.txt");
      nnet::load_weights_from_txt<weight96_t, 65536>(w96, "w96.txt");
      //hls-fpga-machine-learning insert top-level-function
      myproject(em_barrel,scalars,tracks,layer102_out,w25,w30,w38,w69,w76,w81,w87,w92,w96);

      if (e % CHECKPOINT == 0) {
        std::cout << "Predictions" << std::endl;
        //hls-fpga-machine-learning insert predictions
        for(int i = 0; i < N_LAYER_100; i++) {
          std::cout << pr[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Quantized predictions" << std::endl;
        //hls-fpga-machine-learning insert quantized
        nnet::print_result_s<result_t, N_LAYER_100>(layer102_out, std::cout, true);
      }
      e++;

      //hls-fpga-machine-learning insert tb-output
      nnet::print_result_s<result_t, N_LAYER_100>(layer102_out, fout);

    }
    fin.close();
    fpr.close();
  }

  fout.close();
  std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

  return 0;
}
