#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>
#include <sparkit/data/dense_hessenberg_eigen.hpp>
#include <sparkit/data/eigen_target.hpp>

namespace sparkit::data::detail {

  using size_type = sparkit::config::size_type;

  /**
   * @brief Configuration for the Arnoldi eigensolver.
   */
  template <typename T>
  struct Arnoldi_eig_config {
    size_type num_eigenvalues{};
    size_type krylov_dimension{};
    T tolerance{};
    size_type max_restarts{};
    Eigen_target target{};
    bool collect_residuals{};
  };

  /**
   * @brief Result of the Arnoldi eigensolver.
   *
   * Eigenvalues are split into real/imaginary parts. Complex
   * eigenvalues of real matrices appear as conjugate pairs.
   */
  template <typename T>
  struct Arnoldi_eig_result {
    std::vector<T> eigenvalues_real{};
    std::vector<T> eigenvalues_imag{};
    std::vector<std::vector<T>> eigenvectors{};
    std::vector<T> residual_norms{};
    size_type computed_restarts{};
    size_type converged_count{};
    bool converged{};
  };

  /**
   * @brief Implicitly Restarted Arnoldi eigensolver for general
   *        (nonsymmetric) matrices.
   *
   * Computes a few eigenvalues/eigenvectors of a large sparse matrix
   * through the matrix-free callable interface. Uses Francis
   * double-shift QR on the Hessenberg matrix for implicit restarts,
   * keeping all arithmetic real.
   *
   * Reference: Lehoucq, Sorensen, Yang, "ARPACK Users' Guide" (1998).
   *
   * @tparam T               Value type.
   * @tparam LinearOperator  Callable: op(first, last, output_first).
   */
  template <typename T, typename LinearOperator>
  class ArnoldiEigSolver {
  public:
    ArnoldiEigSolver(
      size_type n, Arnoldi_eig_config<T> cfg, LinearOperator linear_operator)
        : n_{n}
        , cfg_{cfg}
        , linear_operator_{linear_operator}
        , nev_{cfg.num_eigenvalues}
        , m_{cfg.krylov_dimension} {
      init();
      run();
    }

    Arnoldi_eig_result<T>
    result() const {
      return result_;
    }

  private:
    size_type n_{};
    Arnoldi_eig_config<T> cfg_{};
    LinearOperator linear_operator_;
    size_type nev_{};
    size_type m_{};

    // Arnoldi basis V_[0..m], each of size n.
    std::vector<std::vector<T>> V_{};
    // Upper Hessenberg matrix H_, (m+1)×m stored row-major in flat
    // vector. H_[i*m + j] = H(i, j).
    std::vector<T> H_{};
    // Workspace.
    std::vector<T> w_{};

    Arnoldi_eig_result<T> result_{};

    T&
    h(size_type i, size_type j) {
      return H_[static_cast<std::size_t>(i * (m_ + 1) + j)];
    }

    T
    h(size_type i, size_type j) const {
      return H_[static_cast<std::size_t>(i * (m_ + 1) + j)];
    }

    void
    init() {
      if (m_ > n_) { m_ = n_; }
      if (nev_ > m_) { nev_ = m_; }

      auto sn = static_cast<std::size_t>(n_);
      auto sm = static_cast<std::size_t>(m_);

      V_.resize(sm + 1);
      for (auto& v : V_) {
        v.assign(sn, T{0});
      }
      H_.assign((sm + 1) * (sm + 1), T{0});
      w_.assign(sn, T{0});

      // Random unit starting vector.
      std::mt19937 rng{42};
      std::uniform_real_distribution<T> dist{T{-1}, T{1}};
      for (std::size_t i = 0; i < sn; ++i) {
        V_[0][i] = dist(rng);
      }
      normalize(V_[0]);
    }

    void
    run() {
      arnoldi_steps(0, m_);

      for (size_type restart = 0; restart < cfg_.max_restarts; ++restart) {
        result_.computed_restarts = restart + 1;

        auto eig = eigendecompose_hessenberg();

        auto wanted = select_wanted(eig.eigenvalues_real, eig.eigenvalues_imag);

        if (check_convergence(eig, wanted) || nev_ == m_) {
          extract_ritz_pairs(eig, wanted);
          result_.converged = true;
          return;
        }

        implicit_restart(eig, wanted);
      }

      // Max restarts exhausted.
      auto eig = eigendecompose_hessenberg();
      auto wanted = select_wanted(eig.eigenvalues_real, eig.eigenvalues_imag);
      extract_ritz_pairs(eig, wanted);
      result_.converged = false;
    }

    // Perform Arnoldi steps from index start to stop-1.
    void
    arnoldi_steps(size_type start, size_type stop) {
      for (size_type j = start; j < stop; ++j) {
        auto sj = static_cast<std::size_t>(j);
        auto sn = static_cast<std::size_t>(n_);

        // w = A * v_j.
        linear_operator_(V_[sj].data(), V_[sj].data() + n_, w_.data());

        // Modified Gram-Schmidt orthogonalization.
        for (size_type i = 0; i <= j; ++i) {
          T hij = dot(V_[static_cast<std::size_t>(i)], w_);
          h(i, j) = hij;
          for (std::size_t k = 0; k < sn; ++k) {
            w_[k] -= hij * V_[static_cast<std::size_t>(i)][k];
          }
        }
        // Second MGS pass for stability.
        for (size_type i = 0; i <= j; ++i) {
          T hij = dot(V_[static_cast<std::size_t>(i)], w_);
          h(i, j) += hij;
          for (std::size_t k = 0; k < sn; ++k) {
            w_[k] -= hij * V_[static_cast<std::size_t>(i)][k];
          }
        }

        T h_jp1_j = norm(w_);
        h(j + 1, j) = h_jp1_j;

        if (h_jp1_j > T{0}) {
          for (std::size_t k = 0; k < sn; ++k) {
            V_[sj + 1][k] = w_[k] / h_jp1_j;
          }
        } else {
          restart_with_random_vector(j);
        }
      }
    }

    void
    restart_with_random_vector(size_type j) {
      auto sj = static_cast<std::size_t>(j);
      auto sn = static_cast<std::size_t>(n_);
      std::mt19937 rng{static_cast<unsigned>(j + 137)};
      std::uniform_real_distribution<T> dist{T{-1}, T{1}};
      for (std::size_t i = 0; i < sn; ++i) {
        V_[sj + 1][i] = dist(rng);
      }
      for (size_type k = 0; k <= j; ++k) {
        T hh = dot(V_[static_cast<std::size_t>(k)], V_[sj + 1]);
        for (std::size_t i = 0; i < sn; ++i) {
          V_[sj + 1][i] -= hh * V_[static_cast<std::size_t>(k)][i];
        }
      }
      normalize(V_[sj + 1]);
    }

    // Extract the m×m leading submatrix of H and compute its
    // eigenvalues.
    Hessenberg_eigen_result<T>
    eigendecompose_hessenberg() const {
      // Copy the m×m leading submatrix into a contiguous array.
      auto sm = static_cast<std::size_t>(m_);
      std::vector<T> Hm(sm * sm, T{0});
      for (size_type i = 0; i < m_; ++i) {
        for (size_type j = 0; j < m_; ++j) {
          Hm[static_cast<std::size_t>(i * m_ + j)] = h(i, j);
        }
      }

      return hessenberg_eigen(
        std::move(Hm),
        m_,
        Hessenberg_eigen_config<T>{
          .tolerance = T{1e-15}, .max_iterations = m_ * 30});
    }

    // Select wanted eigenvalue indices based on target.
    std::vector<size_type>
    select_wanted(
      std::vector<T> const& evals_real,
      std::vector<T> const& evals_imag) const {
      auto count = static_cast<std::size_t>(m_);
      std::vector<size_type> order(count);
      std::iota(order.begin(), order.end(), size_type{0});

      switch (cfg_.target) {
        case Eigen_target::largest_magnitude:
          std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            auto sa = static_cast<std::size_t>(a);
            auto sb = static_cast<std::size_t>(b);
            return (evals_real[sa] * evals_real[sa] +
                    evals_imag[sa] * evals_imag[sa]) >
                   (evals_real[sb] * evals_real[sb] +
                    evals_imag[sb] * evals_imag[sb]);
          });
          break;
        case Eigen_target::smallest_magnitude:
          std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            auto sa = static_cast<std::size_t>(a);
            auto sb = static_cast<std::size_t>(b);
            return (evals_real[sa] * evals_real[sa] +
                    evals_imag[sa] * evals_imag[sa]) <
                   (evals_real[sb] * evals_real[sb] +
                    evals_imag[sb] * evals_imag[sb]);
          });
          break;
        case Eigen_target::largest_algebraic:
        case Eigen_target::largest_real:
          std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return evals_real[static_cast<std::size_t>(a)] >
                   evals_real[static_cast<std::size_t>(b)];
          });
          break;
        case Eigen_target::smallest_algebraic:
        case Eigen_target::smallest_real:
          std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return evals_real[static_cast<std::size_t>(a)] <
                   evals_real[static_cast<std::size_t>(b)];
          });
          break;
      }

      order.resize(static_cast<std::size_t>(nev_));

      // Ensure complex conjugate pairs are kept together.
      ensure_conjugate_pairs(order, evals_real, evals_imag);

      return order;
    }

    // If a complex eigenvalue is selected, ensure its conjugate is
    // too.
    static void
    ensure_conjugate_pairs(
      std::vector<size_type>& order,
      std::vector<T> const& evals_real,
      std::vector<T> const& evals_imag) {
      std::vector<bool> selected(evals_real.size(), false);
      for (auto idx : order) {
        selected[static_cast<std::size_t>(idx)] = true;
      }

      for (std::size_t i = 0; i < order.size(); ++i) {
        auto idx = static_cast<std::size_t>(order[i]);
        if (std::abs(evals_imag[idx]) < T{1e-20}) { continue; }
        // Find conjugate.
        for (std::size_t j = 0; j < evals_real.size(); ++j) {
          if (j == idx || selected[j]) { continue; }
          if (
            std::abs(evals_real[j] - evals_real[idx]) < T{1e-10} &&
            std::abs(evals_imag[j] + evals_imag[idx]) < T{1e-10}) {
            order.push_back(static_cast<size_type>(j));
            selected[j] = true;
            break;
          }
        }
      }
    }

    bool
    check_convergence(
      Hessenberg_eigen_result<T> const& eig,
      std::vector<size_type> const& wanted) {
      bool all_converged = true;
      result_.residual_norms.clear();
      result_.converged_count = 0;

      T h_mp1_m = h(m_, m_ - 1);

      // For Arnoldi: residual for Ritz pair θ_i is
      // |h_{m+1,m} * e_m^T * y_i|
      // where y_i is the eigenvector of H_m for θ_i.
      // Since we don't compute eigenvectors of H in the dense
      // solver, we use the simpler bound: |h_{m+1,m}| as a
      // global residual estimate.
      for (auto idx : wanted) {
        auto si = static_cast<std::size_t>(idx);
        T mag = std::sqrt(
          eig.eigenvalues_real[si] * eig.eigenvalues_real[si] +
          eig.eigenvalues_imag[si] * eig.eigenvalues_imag[si]);
        T abs_res = std::abs(h_mp1_m);
        T threshold = cfg_.tolerance * std::max(mag, T{1e-20});
        result_.residual_norms.push_back(abs_res);
        if (abs_res <= threshold) {
          ++result_.converged_count;
        } else {
          all_converged = false;
        }
      }

      return all_converged;
    }

    // Implicit restart: apply unwanted shifts via Francis QR on H,
    // then compress basis and continue.
    void
    implicit_restart(
      Hessenberg_eigen_result<T> const& eig,
      std::vector<size_type> const& wanted) {
      // Determine unwanted eigenvalues.
      std::vector<bool> is_wanted(static_cast<std::size_t>(m_), false);
      for (auto idx : wanted) {
        is_wanted[static_cast<std::size_t>(idx)] = true;
      }

      // Save residual before shifts.
      T h_mp1_m = h(m_, m_ - 1);
      auto v_mp1 = V_[static_cast<std::size_t>(m_)];

      // Extract the m×m Hessenberg into contiguous storage for QR
      // shifts.
      auto sm = static_cast<std::size_t>(m_);
      std::vector<T> Hm(sm * sm, T{0});
      for (size_type i = 0; i < m_; ++i) {
        for (size_type j = 0; j < m_; ++j) {
          Hm[static_cast<std::size_t>(i * m_ + j)] = h(i, j);
        }
      }

      // Track Q_total as an explicit m×m matrix (identity initially).
      std::vector<T> Q(sm * sm, T{0});
      for (size_type i = 0; i < m_; ++i) {
        Q[static_cast<std::size_t>(i * m_ + i)] = T{1};
      }

      // Apply shifts for each unwanted eigenvalue.
      // For real unwanted: single-shift QR step.
      // For complex conjugate pair: double-shift.
      std::vector<bool> processed(static_cast<std::size_t>(m_), false);

      for (size_type idx = 0; idx < m_; ++idx) {
        auto si = static_cast<std::size_t>(idx);
        if (is_wanted[si] || processed[si]) { continue; }
        processed[si] = true;

        T re = eig.eigenvalues_real[si];
        T im = eig.eigenvalues_imag[si];

        if (std::abs(im) < T{1e-14}) {
          // Real shift.
          apply_single_shift_qr(Hm, Q, m_, re);
        } else {
          // Double shift (complex conjugate pair).
          // Find and mark the conjugate.
          for (size_type j = idx + 1; j < m_; ++j) {
            auto sj = static_cast<std::size_t>(j);
            if (
              !is_wanted[sj] && !processed[sj] &&
              std::abs(eig.eigenvalues_real[sj] - re) < T{1e-10} &&
              std::abs(eig.eigenvalues_imag[sj] + im) < T{1e-10}) {
              processed[sj] = true;
              break;
            }
          }
          T s = T{2} * re;         // trace of 2×2 block
          T t = re * re + im * im; // det of 2×2 block
          apply_double_shift_qr(Hm, Q, m_, s, t);
        }
      }

      // Update V and H from the shifted factorization.
      // V_new = V_old * Q.
      auto sn = static_cast<std::size_t>(n_);
      auto snev = static_cast<std::size_t>(nev_);

      std::vector<std::vector<T>> V_new(sm + 1);
      for (std::size_t j = 0; j <= sm; ++j) {
        V_new[j].assign(sn, T{0});
      }
      for (std::size_t j = 0; j < sm; ++j) {
        for (std::size_t i = 0; i < sm; ++i) {
          T q_ij = Q[i * sm + j];
          for (std::size_t k = 0; k < sn; ++k) {
            V_new[j][k] += q_ij * V_[i][k];
          }
        }
      }

      // Copy H back.
      for (size_type i = 0; i < m_; ++i) {
        for (size_type j = 0; j < m_; ++j) {
          h(i, j) = Hm[static_cast<std::size_t>(i * m_ + j)];
        }
      }

      // Residual update: new v_{k+1} contribution.
      T sigma_k = h(nev_, nev_ - 1);
      T tau = h_mp1_m * Q[(sm - 1) * sm + snev - 1];

      for (std::size_t i = 0; i < sn; ++i) {
        V_new[snev][i] = sigma_k * V_new[snev][i] + tau * v_mp1[i];
      }
      T beta_new = norm(V_new[snev]);
      if (beta_new > T{0}) {
        for (std::size_t i = 0; i < sn; ++i) {
          V_new[snev][i] /= beta_new;
        }
      }
      h(nev_, nev_ - 1) = beta_new;

      // Copy V_new back.
      for (std::size_t j = 0; j <= sm; ++j) {
        V_[j] = std::move(V_new[j]);
      }

      // Continue Arnoldi from step nev to m.
      arnoldi_steps(nev_, m_);
    }

    // Single-shift QR step on the m×m Hessenberg Hm, accumulating
    // rotations in Q.
    static void
    apply_single_shift_qr(
      std::vector<T>& Hm, std::vector<T>& Q, size_type m, T mu) {
      auto hm = [&](size_type i, size_type j) -> T& {
        return Hm[static_cast<std::size_t>(i * m + j)];
      };
      auto q = [&](size_type i, size_type j) -> T& {
        return Q[static_cast<std::size_t>(i * m + j)];
      };

      for (size_type k = 0; k < m - 1; ++k) {
        T x = hm(k, k) - mu;
        T z = hm(k + 1, k);

        T r = std::sqrt(x * x + z * z);
        T cs{}, sn{};
        if (r == T{0}) {
          cs = T{1};
          sn = T{0};
        } else {
          cs = x / r;
          sn = z / r;
        }

        // Apply from left: rows k, k+1.
        for (size_type j = 0; j < m; ++j) {
          T a = hm(k, j);
          T b = hm(k + 1, j);
          hm(k, j) = cs * a + sn * b;
          hm(k + 1, j) = -sn * a + cs * b;
        }

        // Apply from right: columns k, k+1.
        size_type row_end = std::min(k + 3, m);
        for (size_type i = 0; i < row_end; ++i) {
          T a = hm(i, k);
          T b = hm(i, k + 1);
          hm(i, k) = cs * a + sn * b;
          hm(i, k + 1) = -sn * a + cs * b;
        }

        // Accumulate in Q.
        for (size_type i = 0; i < m; ++i) {
          T a = q(i, k);
          T b = q(i, k + 1);
          q(i, k) = cs * a + sn * b;
          q(i, k + 1) = -sn * a + cs * b;
        }
      }
    }

    // Double-shift QR step (Francis) on the m×m Hessenberg Hm.
    static void
    apply_double_shift_qr(
      std::vector<T>& Hm, std::vector<T>& Q, size_type m, T s, T t) {
      auto hm = [&](size_type i, size_type j) -> T& {
        return Hm[static_cast<std::size_t>(i * m + j)];
      };
      auto q = [&](size_type i, size_type j) -> T& {
        return Q[static_cast<std::size_t>(i * m + j)];
      };

      // First column of M = H² - s*H + t*I.
      T x = hm(0, 0) * hm(0, 0) + hm(0, 1) * hm(1, 0) - s * hm(0, 0) + t;
      T y = hm(1, 0) * (hm(0, 0) + hm(1, 1) - s);
      T z = (m > 2) ? hm(1, 0) * hm(2, 1) : T{0};

      for (size_type k = 0; k <= m - 3; ++k) {
        size_type r = std::min(k + 3, m - 1);
        size_type bulge = r - k + 1;

        T v0{}, v1{}, v2{};
        T beta_h{};
        householder_3(x, y, z, bulge, v0, v1, v2, beta_h);

        // Apply from left.
        for (size_type j = (k > 0 ? k - 1 : size_type{0}); j < m; ++j) {
          T dot_val = v0 * hm(k, j);
          if (bulge >= 2) { dot_val += v1 * hm(k + 1, j); }
          if (bulge >= 3) { dot_val += v2 * hm(k + 2, j); }
          dot_val *= beta_h;
          hm(k, j) -= v0 * dot_val;
          if (bulge >= 2) { hm(k + 1, j) -= v1 * dot_val; }
          if (bulge >= 3) { hm(k + 2, j) -= v2 * dot_val; }
        }

        // Apply from right.
        size_type row_end = std::min(r + 2, m);
        for (size_type i = 0; i < row_end; ++i) {
          T dot_val = v0 * hm(i, k);
          if (bulge >= 2) { dot_val += v1 * hm(i, k + 1); }
          if (bulge >= 3) { dot_val += v2 * hm(i, k + 2); }
          dot_val *= beta_h;
          hm(i, k) -= v0 * dot_val;
          if (bulge >= 2) { hm(i, k + 1) -= v1 * dot_val; }
          if (bulge >= 3) { hm(i, k + 2) -= v2 * dot_val; }
        }

        // Accumulate in Q.
        for (size_type i = 0; i < m; ++i) {
          T dot_val = v0 * q(i, k);
          if (bulge >= 2) { dot_val += v1 * q(i, k + 1); }
          if (bulge >= 3) { dot_val += v2 * q(i, k + 2); }
          dot_val *= beta_h;
          q(i, k) -= v0 * dot_val;
          if (bulge >= 2) { q(i, k + 1) -= v1 * dot_val; }
          if (bulge >= 3) { q(i, k + 2) -= v2 * dot_val; }
        }

        // Next bulge.
        if (k + 3 < m) {
          x = hm(k + 1, k);
          y = hm(k + 2, k);
          z = (k + 3 < m - 1) ? hm(k + 3, k) : T{0};
        }
      }

      // Final 2×2 Givens.
      {
        size_type k = m - 2;
        T x_val = hm(k, k > 0 ? k - 1 : size_type{0});
        T z_val = hm(k + 1, k > 0 ? k - 1 : size_type{0});
        T r_val = std::sqrt(x_val * x_val + z_val * z_val);
        if (r_val > T{0}) {
          T cs = x_val / r_val;
          T sn = z_val / r_val;

          for (size_type j = 0; j < m; ++j) {
            T a = hm(k, j);
            T b = hm(k + 1, j);
            hm(k, j) = cs * a + sn * b;
            hm(k + 1, j) = -sn * a + cs * b;
          }
          for (size_type i = 0; i < m; ++i) {
            T a = hm(i, k);
            T b = hm(i, k + 1);
            hm(i, k) = cs * a + sn * b;
            hm(i, k + 1) = -sn * a + cs * b;
          }
          for (size_type i = 0; i < m; ++i) {
            T a = q(i, k);
            T b = q(i, k + 1);
            q(i, k) = cs * a + sn * b;
            q(i, k + 1) = -sn * a + cs * b;
          }
        }
      }

      // Clean up below-Hessenberg entries.
      for (size_type i = 2; i < m; ++i) {
        hm(i, i - 2) = T{0};
        if (i > 2) { hm(i, i - 3) = T{0}; }
      }
    }

    // Extract Ritz pairs for wanted eigenvalues.
    void
    extract_ritz_pairs(
      Hessenberg_eigen_result<T> const& eig,
      std::vector<size_type> const& wanted) {
      result_.eigenvalues_real.clear();
      result_.eigenvalues_imag.clear();
      result_.eigenvectors.clear();

      auto sn = static_cast<std::size_t>(n_);

      // Compute eigenvectors of H_m for each wanted eigenvalue.
      // For real eigenvalues: solve (H - θI)y ≈ 0 via inverse
      // iteration. Simplified: use the Arnoldi relation directly.
      for (auto idx : wanted) {
        auto si = static_cast<std::size_t>(idx);
        result_.eigenvalues_real.push_back(eig.eigenvalues_real[si]);
        result_.eigenvalues_imag.push_back(eig.eigenvalues_imag[si]);

        // For eigenvector: compute y = eigenvector of H_m.
        auto y = hessenberg_eigenvector(
          eig.eigenvalues_real[si], eig.eigenvalues_imag[si]);

        // Ritz vector: x = V_m * y.
        std::vector<T> x(sn, T{0});
        for (size_type j = 0; j < m_; ++j) {
          auto sj = static_cast<std::size_t>(j);
          for (std::size_t i = 0; i < sn; ++i) {
            x[i] += y[sj] * V_[sj][i];
          }
        }
        normalize(x);
        result_.eigenvectors.push_back(std::move(x));
      }
    }

    // Compute eigenvector of H_m for eigenvalue (re, im) via inverse
    // iteration.
    std::vector<T>
    hessenberg_eigenvector(T re, T im) const {
      auto sm = static_cast<std::size_t>(m_);
      std::vector<T> y(sm, T{0});

      if (std::abs(im) < T{1e-14}) {
        // Real eigenvalue: inverse iteration.
        // (H - re*I) * y ≈ 0. Start with random y, solve
        // iteratively.
        std::mt19937 rng{42};
        std::uniform_real_distribution<T> dist{T{-1}, T{1}};
        for (std::size_t i = 0; i < sm; ++i) {
          y[i] = dist(rng);
        }

        T shift = re;
        for (int iter = 0; iter < 3; ++iter) {
          // Solve (H - shift*I) * z = y via back substitution on
          // the Hessenberg (with LU-like approach).
          auto z = solve_shifted_hessenberg(shift, y);
          T nrm = T{0};
          for (auto v : z) {
            nrm += v * v;
          }
          nrm = std::sqrt(nrm);
          if (nrm > T{0}) {
            for (auto& v : z) {
              v /= nrm;
            }
          }
          y = z;
        }
      } else {
        // Complex eigenvalue: just use the real part of the
        // eigenvector. Use a single inverse iteration step.
        std::mt19937 rng{43};
        std::uniform_real_distribution<T> dist{T{-1}, T{1}};
        for (std::size_t i = 0; i < sm; ++i) {
          y[i] = dist(rng);
        }
        auto z = solve_shifted_hessenberg(re, y);
        T nrm = T{0};
        for (auto v : z) {
          nrm += v * v;
        }
        nrm = std::sqrt(nrm);
        if (nrm > T{0}) {
          for (auto& v : z) {
            v /= nrm;
          }
        }
        y = z;
      }

      return y;
    }

    // Solve (H_m - shift*I) * z = b using Gaussian elimination with
    // partial pivoting on the Hessenberg system.
    std::vector<T>
    solve_shifted_hessenberg(T shift, std::vector<T> const& b) const {
      auto sm = static_cast<std::size_t>(m_);
      // Copy H_m and shift.
      std::vector<T> L(sm * sm, T{0});
      for (size_type i = 0; i < m_; ++i) {
        for (size_type j = 0; j < m_; ++j) {
          L[static_cast<std::size_t>(i * m_ + j)] = h(i, j);
        }
        L[static_cast<std::size_t>(i * m_ + i)] -= shift;
      }

      std::vector<T> rhs = b;

      // Forward elimination with partial pivoting.
      for (size_type k = 0; k < m_ - 1; ++k) {
        auto sk = static_cast<std::size_t>(k);
        auto sk1 = static_cast<std::size_t>(k + 1);

        // Only need to check rows k and k+1 (Hessenberg).
        if (std::abs(L[sk1 * sm + sk]) > std::abs(L[sk * sm + sk])) {
          // Swap rows k and k+1.
          for (size_type j = 0; j < m_; ++j) {
            auto sj = static_cast<std::size_t>(j);
            std::swap(L[sk * sm + sj], L[sk1 * sm + sj]);
          }
          std::swap(rhs[sk], rhs[sk1]);
        }

        T pivot = L[sk * sm + sk];
        if (std::abs(pivot) < T{1e-30}) {
          // Near-singular: perturb.
          L[sk * sm + sk] = T{1e-30};
          pivot = T{1e-30};
        }
        T factor = L[sk1 * sm + sk] / pivot;
        for (size_type j = k; j < m_; ++j) {
          auto sj = static_cast<std::size_t>(j);
          L[sk1 * sm + sj] -= factor * L[sk * sm + sj];
        }
        rhs[sk1] -= factor * rhs[sk];
      }

      // Fix last pivot if needed.
      {
        auto sk = static_cast<std::size_t>(m_ - 1);
        if (std::abs(L[sk * sm + sk]) < T{1e-30}) {
          L[sk * sm + sk] = T{1e-30};
        }
      }

      // Back substitution.
      std::vector<T> z(sm, T{0});
      for (size_type i = m_ - 1; i >= 0; --i) {
        auto si = static_cast<std::size_t>(i);
        T sum = rhs[si];
        for (size_type j = i + 1; j < m_; ++j) {
          auto sj = static_cast<std::size_t>(j);
          sum -= L[si * sm + sj] * z[sj];
        }
        z[si] = sum / L[si * sm + si];
        if (i == 0) { break; }
      }

      return z;
    }

    static T
    dot(std::vector<T> const& a, std::vector<T> const& b) {
      return std::inner_product(a.begin(), a.end(), b.begin(), T{0});
    }

    static T
    norm(std::vector<T> const& v) {
      using std::sqrt;
      return sqrt(dot(v, v));
    }

    static void
    normalize(std::vector<T>& v) {
      T n = norm(v);
      if (n > T{0}) {
        for (auto& x : v) {
          x /= n;
        }
      }
    }
  };

  /**
   * @brief Convenience wrapper for the Arnoldi eigensolver.
   */
  template <typename T, typename LinearOperator>
  Arnoldi_eig_result<T>
  arnoldi_eig(
    size_type n, Arnoldi_eig_config<T> cfg, LinearOperator linear_operator) {
    auto solver = ArnoldiEigSolver<T, LinearOperator>(n, cfg, linear_operator);
    return solver.result();
  }

} // end of namespace sparkit::data::detail
