#include "Image.hpp"

imgutils::Image NLBstep1(const imgutils::Image &noisy, float sigma,
                         int nthreads = 0);
imgutils::Image NLBstep2(const imgutils::Image &noisy,
                         const imgutils::Image &guide,
                         float sigma, int nthreads = 0);

void set_patch_size1(int);
void set_search_window1(int);
void set_nsim1(int);
void set_flat_threshold1(float);
void print_params1(void);

void set_patch_size2(int);
void set_search_window2(int);
void set_nsim_min2(int);
void set_tau2(float);
void print_params2(void);
