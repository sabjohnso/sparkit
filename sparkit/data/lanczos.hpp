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
#include <sparkit/data/dense_tridiagonal_eigen.hpp>
#include <sparkit/data/eigen_target.hpp>

namespace sparkit::data::detail {

  using size_type = sparkit::config::size_type;

  /**
   * @brief Configuration for the Lanczos eigensolver.
   */
  template <typename T>
  struct Lanczos_config {
    size_type num_eigenvalues{};
    size_type krylov_dimension{};
    T tolerance{};
    size_type max_restarts{};
    Eigen_target target{};
    bool collect_residuals{};
  };

  /**
   * @brief Result of the Lanczos eigensolver.
   */
  template <typename T>
  struct Lanczos_result {
    std::vector<T> eigenvalues{};
    std::vector<std::vector<T>> eigenvectors{};
    std::vector<T> residual_norms{};
    size_type computed_restarts{};
    size_type converged_count{};
    bool converged{};
    std::vector<T> restart_residuals{};
  };

  /**
   * @brief Implicitly Restarted Lanczos eigensolver for symmetric
   *        matrices.
   *
   * Computes a few eigenvalues/eigenvectors of a large sparse
   * symmetric matrix through the matrix-free callable interface.
   *
   * Reference: Sorensen (1992), "Implicit Application of Polynomial
   *   Filters in a k-Step Arnoldi Method"; Lehoucq, Sorensen, Yang,
   *   "ARPACK Users' Guide" (1998).
   *
   * @tparam T               Value type.
   * @tparam LinearOperator  Callable: op(first, last, output_first).
   */
  template <typename T, typename LinearOperator>
  class LanczosSolver {
  public:
    LanczosSolver(
      size_type n, Lanczos_config<T> cfg, LinearOperator linear_operator)
        : n_{n}
        , cfg_{cfg}
        , linear_operator_{linear_operator}
        , nev_{cfg.num_eigenvalues}
        , m_{cfg.krylov_dimension} {
      init();
      run();
    }

    Lanczos_result<T>
    result() const {
      return result_;
    }

  private:
    size_type n_{};
    Lanczos_config<T> cfg_{};
    LinearOperator linear_operator_;
    size_type nev_{};
    size_type m_{};

    // Lanczos basis V_[0..m], each of size n.
    std::vector<std::vector<T>> V_{};
    // Tridiagonal: alpha (diagonal, size m), beta (subdiagonal, size
    // m+1).
    std::vector<T> alpha_{};
    std::vector<T> beta_{};
    // Workspace.
    std::vector<T> w_{};

    Lanczos_result<T> result_{};

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
      alpha_.assign(sm, T{0});
      beta_.assign(sm + 1, T{0});
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
      lanczos_steps(0, m_);

      for (size_type restart = 0; restart < cfg_.max_restarts; ++restart) {
        result_.computed_restarts = restart + 1;

        auto tri = eigendecompose_tridiagonal();
        auto wanted = select_wanted(tri.eigenvalues);

        if (check_convergence(tri, wanted) || nev_ == m_) {
          extract_ritz_pairs(tri, wanted);
          result_.converged = true;
          return;
        }

        implicit_restart(tri, wanted);
      }

      // Max restarts exhausted.
      auto tri = eigendecompose_tridiagonal();
      auto wanted = select_wanted(tri.eigenvalues);
      extract_ritz_pairs(tri, wanted);
      result_.converged = false;
    }

    Tridiagonal_eigen_result<T>
    eigendecompose_tridiagonal() const {
      return tridiagonal_eigen(
        std::vector<T>(alpha_.begin(), alpha_.begin() + m_),
        std::vector<T>(beta_.begin() + 1, beta_.begin() + m_),
        Tridiagonal_eigen_config<T>{
          .tolerance = T{1e-15}, .max_iterations = m_ * 30},
        true);
    }

    bool
    check_convergence(
      Tridiagonal_eigen_result<T> const& tri,
      std::vector<size_type> const& wanted) {
      bool all_converged = true;
      result_.residual_norms.clear();
      result_.converged_count = 0;

      T beta_m = beta_[static_cast<std::size_t>(m_)];

      for (auto idx : wanted) {
        auto si = static_cast<std::size_t>(idx);
        T abs_res = std::abs(
          beta_m * tri.eigenvectors[si][static_cast<std::size_t>(m_ - 1)]);
        T abs_theta = std::abs(tri.eigenvalues[si]);
        T threshold = cfg_.tolerance * std::max(abs_theta, T{1e-20});
        result_.residual_norms.push_back(abs_res);
        if (abs_res <= threshold) {
          ++result_.converged_count;
        } else {
          all_converged = false;
        }
      }

      if (cfg_.collect_residuals) {
        T max_res = T{0};
        for (auto r : result_.residual_norms) {
          max_res = std::max(max_res, r);
        }
        result_.restart_residuals.push_back(max_res);
      }

      return all_converged;
    }

    // Perform Lanczos steps from index start to stop-1.
    void
    lanczos_steps(size_type start, size_type stop) {
      for (size_type j = start; j < stop; ++j) {
        auto sj = static_cast<std::size_t>(j);
        auto sn = static_cast<std::size_t>(n_);

        // w = A * v_j.
        linear_operator_(V_[sj].data(), V_[sj].data() + n_, w_.data());

        // alpha_j = dot(v_j, w).
        alpha_[sj] = dot(V_[sj], w_);

        // w -= alpha_j * v_j + beta_j * v_{j-1}.
        for (std::size_t i = 0; i < sn; ++i) {
          w_[i] -= alpha_[sj] * V_[sj][i];
        }
        if (j > 0) {
          for (std::size_t i = 0; i < sn; ++i) {
            w_[i] -= beta_[sj] * V_[sj - 1][i];
          }
        }

        // Full reorthogonalization (two-pass MGS).
        reorthogonalize(j);

        // beta_{j+1} = ||w||; v_{j+1} = w / beta_{j+1}.
        T beta_next = norm(w_);
        beta_[static_cast<std::size_t>(j + 1)] = beta_next;

        if (beta_next > T{0}) {
          for (std::size_t i = 0; i < sn; ++i) {
            V_[sj + 1][i] = w_[i] / beta_next;
          }
        } else {
          restart_with_random_vector(j);
        }
      }
    }

    void
    reorthogonalize(size_type j) {
      auto sn = static_cast<std::size_t>(n_);
      for (int pass = 0; pass < 2; ++pass) {
        for (size_type k = 0; k <= j; ++k) {
          T h = dot(V_[static_cast<std::size_t>(k)], w_);
          for (std::size_t i = 0; i < sn; ++i) {
            w_[i] -= h * V_[static_cast<std::size_t>(k)][i];
          }
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
        T h = dot(V_[static_cast<std::size_t>(k)], V_[sj + 1]);
        for (std::size_t i = 0; i < sn; ++i) {
          V_[sj + 1][i] -= h * V_[static_cast<std::size_t>(k)][i];
        }
      }
      normalize(V_[sj + 1]);
    }

    // Select indices of wanted eigenvalues based on target.
    std::vector<size_type>
    select_wanted(std::vector<T> const& eigenvalues) const {
      auto count = static_cast<std::size_t>(m_);
      std::vector<size_type> order(count);
      std::iota(order.begin(), order.end(), size_type{0});

      switch (cfg_.target) {
        case Eigen_target::largest_magnitude:
          std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return std::abs(eigenvalues[static_cast<std::size_t>(a)]) >
                   std::abs(eigenvalues[static_cast<std::size_t>(b)]);
          });
          break;
        case Eigen_target::smallest_magnitude:
          std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return std::abs(eigenvalues[static_cast<std::size_t>(a)]) <
                   std::abs(eigenvalues[static_cast<std::size_t>(b)]);
          });
          break;
        case Eigen_target::largest_algebraic:
        case Eigen_target::largest_real:
          std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return eigenvalues[static_cast<std::size_t>(a)] >
                   eigenvalues[static_cast<std::size_t>(b)];
          });
          break;
        case Eigen_target::smallest_algebraic:
        case Eigen_target::smallest_real:
          std::sort(order.begin(), order.end(), [&](auto a, auto b) {
            return eigenvalues[static_cast<std::size_t>(a)] <
                   eigenvalues[static_cast<std::size_t>(b)];
          });
          break;
      }

      order.resize(static_cast<std::size_t>(nev_));
      return order;
    }

    std::vector<size_type>
    select_unwanted(std::vector<size_type> const& wanted) const {
      std::vector<bool> is_wanted(static_cast<std::size_t>(m_), false);
      for (auto idx : wanted) {
        is_wanted[static_cast<std::size_t>(idx)] = true;
      }

      std::vector<size_type> unwanted;
      for (size_type i = 0; i < m_; ++i) {
        if (!is_wanted[static_cast<std::size_t>(i)]) { unwanted.push_back(i); }
      }
      return unwanted;
    }

    // Implicit restart: apply p = m - nev unwanted shifts, then
    // properly reconstruct the residual and continue Lanczos.
    void
    implicit_restart(
      Tridiagonal_eigen_result<T> const& tri,
      std::vector<size_type> const& wanted) {
      auto unwanted = select_unwanted(wanted);

      // Save the residual norm and vector before shifts.
      T beta_m_saved = beta_[static_cast<std::size_t>(m_)];
      auto v_mp1_saved = V_[static_cast<std::size_t>(m_)];

      // Track the last row of Q_total (= e_{m-1}^T * Q_accumulated).
      // Starts as e_{m-1} and gets updated by each QR step.
      auto sm = static_cast<std::size_t>(m_);
      std::vector<T> q_row(sm, T{0});
      q_row[sm - 1] = T{1};

      // Apply each unwanted eigenvalue as a shift.
      for (auto ui : unwanted) {
        T shift = tri.eigenvalues[static_cast<std::size_t>(ui)];
        tridiagonal_qr_step(shift, q_row);
      }

      // Reconstruct the residual coupling for the compressed
      // factorization. After p shifts on the m-step factorization:
      //   A * V_k = V_k * T_k + f_k * e_k^T
      // where:
      //   f_k = beta_k^+ * v_{k+1}^+ + tau * v_{m+1}
      //   beta_k^+ = T'_{k+1,k} (from shifted tridiagonal)
      //   tau = beta_m * Q_total[m-1, k-1] (tracked via q_row)
      auto snev = static_cast<std::size_t>(nev_);
      auto sn = static_cast<std::size_t>(n_);

      T sigma_k = beta_[snev];
      T tau = beta_m_saved * q_row[snev - 1];

      for (std::size_t i = 0; i < sn; ++i) {
        V_[snev][i] = sigma_k * V_[snev][i] + tau * v_mp1_saved[i];
      }
      T beta_new = norm(V_[snev]);
      if (beta_new > T{0}) {
        for (std::size_t i = 0; i < sn; ++i) {
          V_[snev][i] /= beta_new;
        }
      }
      beta_[snev] = beta_new;

      // Continue Lanczos from step nev to m.
      lanczos_steps(nev_, m_);
    }

    // Apply one implicit QR step with shift mu to the tridiagonal
    // matrix and update the basis V and Q-row tracking.
    void
    tridiagonal_qr_step(T mu, std::vector<T>& q_row) {
      auto sn = static_cast<std::size_t>(n_);

      T x = alpha_[0] - mu;
      T z = beta_[1];

      for (size_type k = 0; k < m_ - 1; ++k) {
        auto sk = static_cast<std::size_t>(k);
        auto sk1 = static_cast<std::size_t>(k + 1);
        auto sk2 = static_cast<std::size_t>(k + 2);

        // Givens rotation to zero z.
        T cs{}, sn_val{};
        T r = std::sqrt(x * x + z * z);
        if (r == T{0}) {
          cs = T{1};
          sn_val = T{0};
        } else {
          cs = x / r;
          sn_val = z / r;
        }

        // Update tridiagonal entries.
        if (k > 0) { beta_[sk] = r; }

        T a_k = alpha_[sk];
        T a_k1 = alpha_[sk1];
        T b_k1 = beta_[sk1];

        alpha_[sk] =
          cs * cs * a_k + T{2} * cs * sn_val * b_k1 + sn_val * sn_val * a_k1;
        alpha_[sk1] =
          sn_val * sn_val * a_k - T{2} * cs * sn_val * b_k1 + cs * cs * a_k1;
        beta_[sk1] =
          cs * sn_val * (a_k1 - a_k) + (cs * cs - sn_val * sn_val) * b_k1;

        // Prepare next bulge.
        if (k + 2 < m_) {
          T b_k2 = beta_[sk2];
          x = beta_[sk1];
          z = sn_val * b_k2;
          beta_[sk2] = cs * b_k2;
        }

        // Update basis vectors: V_new = V * Q.
        for (std::size_t i = 0; i < sn; ++i) {
          T v_k = V_[sk][i];
          T v_k1 = V_[sk1][i];
          V_[sk][i] = cs * v_k + sn_val * v_k1;
          V_[sk1][i] = -sn_val * v_k + cs * v_k1;
        }

        // Track last row of Q_total.
        // q_row * M_k where M_k = G_k^T at block (k, k+1):
        //   q[k]_new = cs * q[k] + sn * q[k+1]
        //   q[k+1]_new = -sn * q[k] + cs * q[k+1]
        T q_k = q_row[sk];
        T q_k1 = q_row[sk1];
        q_row[sk] = cs * q_k + sn_val * q_k1;
        q_row[sk1] = -sn_val * q_k + cs * q_k1;
      }
    }

    // Extract converged Ritz pairs.
    void
    extract_ritz_pairs(
      Tridiagonal_eigen_result<T> const& tri,
      std::vector<size_type> const& wanted) {
      result_.eigenvalues.clear();
      result_.eigenvectors.clear();

      auto sn = static_cast<std::size_t>(n_);

      for (auto idx : wanted) {
        auto si = static_cast<std::size_t>(idx);
        result_.eigenvalues.push_back(tri.eigenvalues[si]);

        // Ritz vector: x_i = V_m * y_i.
        std::vector<T> x(sn, T{0});
        for (size_type j = 0; j < m_; ++j) {
          auto sj = static_cast<std::size_t>(j);
          T coeff = tri.eigenvectors[si][sj];
          for (std::size_t i = 0; i < sn; ++i) {
            x[i] += coeff * V_[sj][i];
          }
        }
        normalize(x);
        result_.eigenvectors.push_back(std::move(x));
      }
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
   * @brief Convenience wrapper for the Lanczos eigensolver.
   *
   * @param n                 Problem dimension.
   * @param cfg               Solver configuration.
   * @param linear_operator   Callable: op(first, last, output_first).
   * @return Lanczos_result with eigenvalues and eigenvectors.
   */
  template <typename T, typename LinearOperator>
  Lanczos_result<T>
  lanczos(size_type n, Lanczos_config<T> cfg, LinearOperator linear_operator) {
    auto solver = LanczosSolver<T, LinearOperator>(n, cfg, linear_operator);
    return solver.result();
  }

} // end of namespace sparkit::data::detail
