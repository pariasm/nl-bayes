#include <cmath>
#include <utility>
#include <tuple>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "Image.hpp"
#include "utils.hpp"
#include "NLBayes.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using imgutils::Image;
using imgutils::BoolMask;
using imgutils::ComputeTiling;
using imgutils::SplitTiles;
using imgutils::MergeTiles;
using std::move;
using std::sqrt;
using std::pair;
using std::abs;
using std::vector;
using std::copy;
using std::nth_element;
using std::max;
using std::swap;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::LDLT;
using Eigen::Success;

namespace step1 {
int patch_size = 3;
int search_window = 8 * patch_size - 1;
int nsim = 50;
float flat_threshold = 1.f;
}

void set_patch_size1(int p) {step1::patch_size = p;}
void set_search_window1(int p) {step1::search_window = p;}
void set_nsim1(int p) {step1::nsim = p;}
void set_flat_threshold1(float p) {step1::flat_threshold = p;}

void print_params1(void) {
  printf("step1::patch_size %d\n", step1::patch_size);
  printf("     ::search_window %d\n", step1::search_window);
  printf("     ::nsim %d\n", step1::nsim);
  printf("     ::flat_threshold %g\n", step1::flat_threshold);
}

namespace step2 {
int patch_size = 5;
int search_window = 8 * patch_size - 1;
int nsim_min = 40;
float tau = 16.f;
}

void set_patch_size2(int p) {step2::patch_size = p;}
void set_search_window2(int p) {step2::search_window = p;}
void set_nsim_min2(int p) {step2::nsim_min = p;}
void set_tau2(float p) {step2::tau = p;}

void print_params2(void) {
  printf("step2::patch_size %d\n", step2::patch_size);
  printf("     ::search_window %d\n", step2::search_window);
  printf("     ::nsim_min %d\n", step2::nsim_min);
  printf("     ::tau %g\n", step2::tau);
}

class PatchDist {
 public:
  float dist;
  int row;
  int col;
  friend bool operator<(const PatchDist& l, const PatchDist& r) {
    return l.dist < r.dist;
  }
};

Image ColorTransform(Image &&src) {
  Image img = move(src);
  if (img.channels() == 3) {
    for (int row = 0; row < img.rows(); ++row) {
      for (int col = 0; col < img.columns(); ++col) {
        float r, g, b;
        r = img.val(col, row, 0);
        g = img.val(col, row, 1);
        b = img.val(col, row, 2);
        img.val(col, row, 0) = (r + g + b) / sqrt(3.f);
        img.val(col, row, 1) = (r - b) / sqrt(2.f);
        img.val(col, row, 2) = (r - 2 * g + b) / sqrt(6.f);
      }
    }
  }
  return img;
}

Image ColorTransformInverse(Image &&src) {
  Image img = move(src);
  if (img.channels() == 3) {
    for (int row = 0; row < img.rows(); ++row) {
      for (int col = 0; col < img.columns(); ++col) {
        float y, u, v;
        y = img.val(col, row, 0);
        u = img.val(col, row, 1);
        v = img.val(col, row, 2);
        img.val(col, row, 0) = (sqrt(2.f) * y + sqrt(3.f) * u + v) / sqrt(6.f);
        img.val(col, row, 1) = (y - sqrt(2.f) * v) / sqrt(3.f);
        img.val(col, row, 2) = (sqrt(2.f) * y - sqrt(3.f) * u + v) / sqrt(6.f);
      }
    }
  }
  return img;
}

namespace step1 {

float ComputeDistance(const Image &src, int r1, int c1, int r2, int c2) {
  float ans = 0.f;
  for (int row = 0; row < patch_size; ++row) {
    for (int col = 0; col < patch_size; ++col) {
      float d = src.val(col + c1, row + r1, 0) - src.val(col + c2, row + r2, 0);
      ans += d * d;
    }
  }
  return ans;
}

pair<Image, Image> compute(const Image &noisy, float sigma) {
  float sigma2 = sigma * sigma;
  int ps2 = patch_size * patch_size;
  int border = (search_window - patch_size) / 2;
  Image result(noisy.rows(), noisy.columns(), noisy.channels());
  Image weights(noisy.rows(), noisy.columns());
  BoolMask processed(noisy.rows(), noisy.columns());

  // (row, col) are the coordinates of the top left pixel of the search window.
  // the central patch starts at (row + border, col + border)
  for (int row = 0; row <= noisy.rows() - search_window; ++row) {
    for (int col = 0; col <= noisy.columns() - search_window; ++col) {
      if (!processed.val(col + border, row + border)) {
        vector<PatchDist> distances;

        // get the group of similar patches
        // patches are indexed via their top left coordinate
        // the central patch is on (row + border, col + border)
        for (int r = row; r <= row + search_window - patch_size; ++r) {
          for (int c = col; c <= col + search_window - patch_size; ++c) {
            float d = ComputeDistance(noisy, r, c, row + border, col + border);
            distances.push_back({d, r, c});
          }
        }
        assert(distances.size() == (2 * border + 1) * (2 * border + 1));
        // we want to be sure to denoise the central patch
        swap(distances[0], distances[border * (2 * border + 2)]);
        // the first nsim element are the smallest ones
        nth_element(distances.begin() + 1,
                    distances.begin() + nsim - 1,
                    distances.end());

        // compute the variance of the block
        float variance = 0.f;
        for (int chan = 0; chan < noisy.channels(); ++chan) {
          float sum = 0.f, sum2 = 0.f;
          for (int i = 0; i < nsim; ++i) {
            for (int r = 0; r < patch_size; ++r) {
              for (int c = 0; c < patch_size; ++c) {
                float val = noisy.val(distances[i].col + c,
                                      distances[i].row + r,
                                      chan);
                sum += val;
                sum2 += val * val;
              }
            }
          }
          variance += (sum2 - (sum * sum / (ps2 * nsim))) /
              ((ps2 * nsim - 1) * noisy.channels());
        }

        for (int chan = 0; chan < noisy.channels(); ++chan) {
          // put all the patches of the group as rows of a matrix
          MatrixXf block(ps2, nsim);
          for (int i = 0; i < nsim; ++i) {
            int pos = 0;
            for (int r = 0; r < patch_size; ++r) {
              for (int c = 0; c < patch_size; ++c) {
                block(pos++, i) = noisy.val(distances[i].col + c,
                                            distances[i].row + r,
                                            chan);
              }
            }
          }

          if (variance < flat_threshold * sigma2) {
            block.setConstant(block.array().mean());
          } else {
            // Compute the centered block
            MatrixXf cblock = block.colwise() - block.rowwise().mean();
            // Compute the covariance matrix of the block of similar patches
            MatrixXf covariance = cblock * cblock.transpose() / (nsim - 1);
            // Bayes' Filtering -> block -= sigma2 C^-1 (block - avg)
            LDLT<MatrixXf> solver(covariance);
            if (solver.info() == Success)
              block -= sigma2 * solver.solve(cblock);
          }

          // Aggregate
          for (int i = 0; i < nsim; ++i) {
            int pos = 0;
            for (int r = 0; r < patch_size; ++r) {
              for (int c = 0; c < patch_size; ++c) {
                result.val(distances[i].col + c, distances[i].row + r, chan) +=
                    block(pos++, i);
              }
            }
          }
        }

        // Mark as done and increment distances
        for (int i = 0; i < nsim; ++i) {
          processed.val(distances[i].col, distances[i].row) = true;
          for (int r = 0; r < patch_size; ++r) {
            for (int c = 0; c < patch_size; ++c) {
              ++weights.val(distances[i].col + c, distances[i].row + r);
            }
          }
        }
      }
    }
  }

  return {move(result), move(weights)};
}
}

namespace step2 {

float ComputeDistance(const Image &src, int r1, int c1, int r2, int c2) {
  float ans = 0.f;
  for (int chan = 0; chan < src.channels(); ++chan) {
    for (int row = 0; row < patch_size; ++row) {
      for (int col = 0; col < patch_size; ++col) {
        float d = src.val(col + c1, row + r1, chan) -
            src.val(col + c2, row + r2, chan);
        ans += d * d;
      }
    }
  }
  return ans;
}

pair<Image, Image> compute(const Image &noisy, const Image &guide, float sigma) {
  float sigma2 = sigma * sigma;
  int ps2 = patch_size * patch_size * noisy.channels();
  int border = (search_window - patch_size) / 2;
  Image result(noisy.rows(), noisy.columns(), noisy.channels());
  Image weights(noisy.rows(), noisy.columns());
  BoolMask processed(noisy.rows(), noisy.columns());

  // (row, col) are the coordinates of the top left pixel of the search window.
  // the central patch starts at (row + border, col + border)
  for (int row = 0; row <= noisy.rows() - search_window; ++row) {
    for (int col = 0; col <= noisy.columns() - search_window; ++col) {
      if (!processed.val(col + border, row + border)) {
        vector<PatchDist> distances;

        // get the group of similar patches
        // patches are indexed via their top left coordinate
        // the central patch is on (row + border, col + border)
        int nsim = 0;
        for (int r = row; r <= row + search_window - patch_size; ++r) {
          for (int c = col; c <= col + search_window - patch_size; ++c) {
            float d = ComputeDistance(guide, r, c, row + border, col + border);
            distances.push_back({d, r, c});
            if (d < tau * ps2)
              ++nsim;
          }
        }
        nsim = max(nsim, nsim_min);
        assert(distances.size() == (2 * border + 1) * (2 * border + 1));
        // we want to be sure to denoise the central patch
        swap(distances[0], distances[border * (2 * border + 2)]);
        // the first nsim element are the smallest ones
        nth_element(distances.begin() + 1, distances.begin() + nsim - 1,
                    distances.end());

        // put all the patches of the group as rows of a matrix
        MatrixXf block(ps2, nsim), gblock(ps2, nsim);
        for (int i = 0; i < nsim; ++i) {
          int pos = 0;
          for (int chan = 0; chan < noisy.channels(); ++chan) {
            for (int r = 0; r < patch_size; ++r) {
              for (int c = 0; c < patch_size; ++c) {
                block(pos, i) = noisy.val(distances[i].col + c,
                                          distances[i].row + r,
                                          chan);
                gblock(pos++, i) = guide.val(distances[i].col + c,
                                             distances[i].row + r,
                                             chan);
              }
            }
          }
        }

        // Compute the mean of the block of similar patches in the guide
        VectorXf mean = gblock.rowwise().mean();
        // Compute the covariance matrix of the block of similar patches in the guide
        MatrixXf covariance = (gblock.colwise() - mean) *
            (gblock.colwise() - mean).transpose() / (nsim - 1);
        // Bayes' Filtering -> block -= sigma2 C^-1 (block - avg)
        LDLT<MatrixXf>
            solver(covariance + sigma2 * MatrixXf::Identity(ps2, ps2));
        if (solver.info() == Success)
          block -= sigma2 * solver.solve(block.colwise() - mean);

        // Aggregate
        for (int i = 0; i < nsim; ++i) {
          int pos = 0;
          for (int chan = 0; chan < noisy.channels(); ++chan) {
            for (int r = 0; r < patch_size; ++r) {
              for (int c = 0; c < patch_size; ++c) {
                result.val(distances[i].col + c, distances[i].row + r, chan) +=
                    block(pos++, i);
              }
            }
          }
        }

        // Mark as done and increment distances
        for (int i = 0; i < nsim; ++i) {
          processed.val(distances[i].col, distances[i].row) = true;
          for (int r = 0; r < patch_size; ++r) {
            for (int c = 0; c < patch_size; ++c) {
              ++weights.val(distances[i].col + c, distances[i].row + r);
            }
          }
        }
      }
    }
  }

  return {move(result), move(weights)};
}
}

Image NLBstep1(const imgutils::Image &noisy, float sigma, int nthreads) {
#ifdef _OPENMP
  if (!nthreads) nthreads = omp_get_max_threads();  // number of threads
#else
  nthreads = 1;
#endif  // _OPENMP

  int offset = step1::search_window / 2;
  pair<int, int> tiling = ComputeTiling(noisy.rows(), noisy.columns(),
                                        nthreads);
  vector<Image> noisy_tiles = SplitTiles(ColorTransform(noisy.copy()),
                                         offset, offset, tiling);
  vector<pair<Image, Image>> result_tiles(nthreads);

#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < nthreads; ++i) {
    result_tiles[i] = step1::compute(noisy_tiles[i], sigma);
  }

  return ColorTransformInverse(MergeTiles(result_tiles, noisy.shape(),
                                          offset, offset, tiling));
}

Image NLBstep2(const imgutils::Image &noisy, const imgutils::Image &guide,
               float sigma, int nthreads) {
#ifdef _OPENMP
  if (!nthreads) nthreads = omp_get_max_threads();  // number of threads
#else
  nthreads = 1;
#endif  // _OPENMP

  int offset = step2::search_window / 2;
  pair<int, int> tiling = ComputeTiling(noisy.rows(), noisy.columns(),
                                        nthreads);
  vector<Image> noisy_tiles = SplitTiles(noisy, offset, offset, tiling);
  vector<Image> guide_tiles = SplitTiles(guide, offset, offset, tiling);
  vector<pair<Image, Image>> result_tiles(nthreads);

#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < nthreads; ++i) {
    result_tiles[i] = step2::compute(noisy_tiles[i], guide_tiles[i], sigma);
  }

  return MergeTiles(result_tiles, noisy.shape(), offset, offset, tiling);
}
