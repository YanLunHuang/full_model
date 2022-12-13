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

#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void myproject(
    hls::stream<input29_t> em_barrel[1], hls::stream<input13_t> scalars[1], hls::stream<input_t> tracks[6],
    hls::stream<result_t> layer102_out[1],
	weight25_t w25[196608],
	weight30_t w30[524288],
	weight38_t w38[1015808],
	weight69_t w69[73728],
	weight76_t w76[147456],
	weight81_t w81[294912],
	weight87_t w87[589824],
	weight92_t w92[589824],
	weight96_t w96[65536]
);

#endif
