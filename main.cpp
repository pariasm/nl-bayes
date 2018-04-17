//
// Created by Nicola Pierazzo on 08/07/16.
//

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include "NLBayes.h"
#include "utils.hpp"

using imgutils::pick_option;
using imgutils::read_image;
using imgutils::save_image;
using imgutils::Image;
using std::cerr;
using std::endl;
using std::move;

int main(int argc, char **argv) {
  const bool usage = static_cast<bool>(pick_option(&argc, argv, "h", nullptr));
  const bool
      no_second_step = static_cast<bool>(pick_option(&argc, argv, "1", NULL));
  const char *second_step_guide = pick_option(&argc, argv, "2", "");
  const bool no_first_step = second_step_guide[0] != '\0';

  const bool verbose = static_cast<bool>(pick_option(&argc, argv, "v", nullptr));

  // check if it is the right call for the algorithm
  if (usage || argc < 2) {
    cerr << "usage: " << argv[0] << " sigma [input [output]] "
         << "[-1 | -2 guide] " << endl;
    return usage ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  if (no_second_step && no_first_step) {
    cerr << "You can't use -1 and -2 together." << endl;
    return EXIT_FAILURE;
  }

#ifndef _OPENMP
  cerr << "Warning: OpenMP not available. The algorithm will run in a single" <<
       " thread." << endl;
#endif

  Image noisy = read_image(argc > 2 ? argv[2] : "-");
  Image guide, result;
  const float sigma = static_cast<float>(atof(argv[1]));

  if (!no_first_step) {
    int ps1 = atoi(pick_option(&argc, argv, "p1", "0")); // patch size
    int sw1 = atoi(pick_option(&argc, argv, "s1", "0")); // search region
    int ns1 = atoi(pick_option(&argc, argv, "n1", "0")); // similar patches
    float ft1 = atof(pick_option(&argc, argv, "f1", "-1")); // flat threshold
    if (ps1 > 0) set_patch_size1(ps1);
    if (sw1 > 0) set_search_window1(sw1);
    if (ns1 > 0) set_nsim1(ns1);
    if (ft1 >= 0) set_flat_threshold1(ft1);
    if (verbose) print_params1();

    guide = NLBstep1(noisy, sigma);
  } else {
    guide = read_image(second_step_guide);
  }

  if (!no_second_step) {
    int ps2 = atoi(pick_option(&argc, argv, "p2", "0")); // patch size
    int sw2 = atoi(pick_option(&argc, argv, "s2", "0")); // search region
    int ns2 = atoi(pick_option(&argc, argv, "n2", "0")); // similar patches
    float dt2 = atof(pick_option(&argc, argv, "d2", "-1")); // dist threshold
    if (ps2 > 0) set_patch_size2(ps2);
    if (sw2 > 0) set_search_window2(sw2);
    if (ns2 > 0) set_nsim_min2(ns2);
    if (dt2 >= 0) set_tau2(dt2);
    if (verbose) print_params2();

    result = NLBstep2(noisy, guide, sigma);
  } else {
    result = move(guide);
  }

  save_image(result, argc > 3 ? argv[3] : "TIFF:-");

  return EXIT_SUCCESS;
}

// vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
