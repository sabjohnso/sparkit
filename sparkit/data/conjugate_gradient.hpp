#pragma once

//
// ... Standard header files
//
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>

//
// ... sparkit header files
//
#include <sparkit/config.hpp>

namespace sparkit::data::detail {

  // Named vector operations for CG kernels.
  //
  // Encapsulates the small linear-algebra primitives that CG needs,
  // expressed with std::transform_reduce / std::transform so the intent
  // is declarative rather than loop-based.

  template <typename T>
  struct Vector_ops {
    static T
    dot(std::span<T const> a, std::span<T const> b) {
      return std::transform_reduce(a.begin(), a.end(), b.begin(), T{0});
    }

    static T
    norm(std::span<T const> v) {
      return std::sqrt(dot(v, v));
    }

    static void
    axpy(T alpha, std::span<T const> x, std::span<T> y) {
      std::transform(x.begin(), x.end(), y.begin(), y.begin(),
                     [alpha](T xi, T yi) { return yi + alpha * xi; });
    }

    static std::vector<T>
    residual(std::span<T const> b, std::span<T const> Ax) {
      std::vector<T> r(b.size());
      std::transform(b.begin(), b.end(), Ax.begin(), r.begin(),
                     [](T bi, T ai) { return bi - ai; });
      return r;
    }

    static void
    update_direction(std::span<T const> z, T beta, std::span<T> p) {
      std::transform(z.begin(), z.end(), p.begin(), p.begin(),
                     [beta](T zi, T pi) { return zi + beta * pi; });
    }
  };

  template <typename T>
  struct Cg_result {
    std::vector<T> x;
    config::size_type iterations;
    T residual_norm;
    bool converged;
  };

  // Unpreconditioned Conjugate Gradient (Templates book, Algorithm 2.3).
  //
  // Solves A*x = b for symmetric positive definite A.
  // apply_A is any callable: std::span<T const> -> std::vector<T>.
  // Convergence criterion: ||r||_2 / ||b||_2 < tolerance.

  template <typename T, typename LinearOperator>
  Cg_result<T>
  conjugate_gradient(LinearOperator apply_A, std::span<T const> b,
                     std::span<T const> x0, T tolerance,
                     config::size_type max_iterations) {
    using Ops = Vector_ops<T>;

    // x = x0
    std::vector<T> x(x0.begin(), x0.end());

    // r = b - A*x0
    auto Ax = apply_A(std::span<T const>{x});
    auto r = Ops::residual(b, std::span<T const>{Ax});

    // b_norm = ||b||_2
    T b_norm = Ops::norm(b);

    if (b_norm == T{0}) { return Cg_result<T>{std::move(x), 0, T{0}, true}; }

    // p = r
    std::vector<T> p(r);

    // rr = r . r
    T rr = Ops::dot(std::span<T const>{r}, std::span<T const>{r});

    for (config::size_type k = 1; k <= max_iterations; ++k) {
      // q = A*p
      auto q = apply_A(std::span<T const>{p});

      // alpha = rr / (p . q)
      T pq = Ops::dot(std::span<T const>{p}, std::span<T const>{q});
      T alpha = rr / pq;

      // x += alpha*p
      Ops::axpy(alpha, std::span<T const>{p}, std::span<T>{x});

      // r -= alpha*q
      Ops::axpy(-alpha, std::span<T const>{q}, std::span<T>{r});

      // rr_new = r . r
      T rr_new = Ops::dot(std::span<T const>{r}, std::span<T const>{r});
      T r_norm = std::sqrt(rr_new);

      if (r_norm / b_norm < tolerance) {
        return Cg_result<T>{std::move(x), k, r_norm, true};
      }

      // beta = rr_new / rr
      T beta = rr_new / rr;

      // p = r + beta*p
      Ops::update_direction(std::span<T const>{r}, beta, std::span<T>{p});

      rr = rr_new;
    }

    T final_norm = std::sqrt(rr);
    return Cg_result<T>{std::move(x), max_iterations, final_norm, false};
  }

  // Preconditioned Conjugate Gradient.
  //
  // apply_A is any callable: std::span<T const> -> std::vector<T>.
  // apply_inv_M is any callable: std::span<T const> -> std::vector<T>
  // that applies M^{-1} to a vector. For IC(0), compose
  // forward_solve + forward_solve_transpose.

  template <typename T, typename LinearOperator, typename Preconditioner>
  Cg_result<T>
  preconditioned_conjugate_gradient(LinearOperator apply_A,
                                    Preconditioner apply_inv_M,
                                    std::span<T const> b, std::span<T const> x0,
                                    T tolerance,
                                    config::size_type max_iterations) {
    using Ops = Vector_ops<T>;

    // x = x0
    std::vector<T> x(x0.begin(), x0.end());

    // r = b - A*x0
    auto Ax = apply_A(std::span<T const>{x});
    auto r = Ops::residual(b, std::span<T const>{Ax});

    // b_norm = ||b||_2
    T b_norm = Ops::norm(b);

    if (b_norm == T{0}) { return Cg_result<T>{std::move(x), 0, T{0}, true}; }

    // z = M^{-1}*r
    auto z = apply_inv_M(std::span<T const>{r});

    // p = z
    std::vector<T> p(z);

    // rz = r . z
    T rz = Ops::dot(std::span<T const>{r}, std::span<T const>{z});

    for (config::size_type k = 1; k <= max_iterations; ++k) {
      // q = A*p
      auto q = apply_A(std::span<T const>{p});

      // alpha = rz / (p . q)
      T pq = Ops::dot(std::span<T const>{p}, std::span<T const>{q});
      T alpha = rz / pq;

      // x += alpha*p
      Ops::axpy(alpha, std::span<T const>{p}, std::span<T>{x});

      // r -= alpha*q
      Ops::axpy(-alpha, std::span<T const>{q}, std::span<T>{r});

      // Check convergence: ||r||_2 / ||b||_2 < tolerance
      T r_norm = Ops::norm(std::span<T const>{r});

      if (r_norm / b_norm < tolerance) {
        return Cg_result<T>{std::move(x), k, r_norm, true};
      }

      // z = M^{-1}*r
      z = apply_inv_M(std::span<T const>{r});

      // rz_new = r . z
      T rz_new = Ops::dot(std::span<T const>{r}, std::span<T const>{z});

      // beta = rz_new / rz
      T beta = rz_new / rz;

      // p = z + beta*p
      Ops::update_direction(std::span<T const>{z}, beta, std::span<T>{p});

      rz = rz_new;
    }

    T final_norm = Ops::norm(std::span<T const>{r});
    return Cg_result<T>{std::move(x), max_iterations, final_norm, false};
  }

} // end of namespace sparkit::data::detail
