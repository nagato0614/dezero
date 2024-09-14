//
// Created by toru on 2024/09/13.
//

#ifndef DEZERO1__NTENSOR_HPP_
#define DEZERO1__NTENSOR_HPP_

#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace nagato::na
{

using Float = float;
template<int rank>
using NTensor = Eigen::Tensor<Float, rank>;

template<int rank>
NTensor<rank>
CreateTensor(const std::vector<std::vector<Float>> &vec,
             const Eigen::array<Eigen::Index, rank> &shape)
{
  NTensor<rank> tensor(shape);

  for (int i = 0; i < shape[0]; ++i)
  {
    for (int j = 0; j < shape[1]; ++j)
    {
      tensor(i, j) = vec[i][j];
    }
  }

  return tensor;
}

template<int rank>
NTensor<rank>
CreateTensor(const std::array<std::array<Float, rank>, rank> &vec,
             const Eigen::array<Eigen::Index, rank> &shape)
{
  NTensor<rank> tensor(shape);

  for (int i = 0; i < shape[0]; ++i)
  {
    for (int j = 0; j < shape[1]; ++j)
    {
      tensor(j, i) = vec[i][j];
    }
  }

  return tensor;
}

/**
 * 任意の次元のテンソルを作成する
 * @tparam rank テンソルの次元
 * @param input_data テンソルに格納するデータ
 * @param shape テンソルの形状を指定するEigen::array
 * @return 作成されたテンソル
 */
template<int rank>
NTensor<rank> CreateTensor(const std::initializer_list<Float> &input_data,
                           const Eigen::array<Eigen::Index, rank> &shape)
{
  // データサイズとテンソルの総要素数が一致するか確認
  if (rank != shape.size())
  {
    throw std::invalid_argument("shapeのサイズとrankが一致しません。");
  }

  // 総要素数を計算
  Eigen::Index total_size = 1;
  for (int i = 0; i < rank; ++i)
  {
    total_size *= shape[i];
  }

  // データサイズとテンソルの総要素数が一致するか確認
  if (static_cast<Eigen::Index>(input_data.size()) != total_size)
  {
    std::cerr << "input : " << input_data.size() << std::endl;
    std::cerr << "total : " << total_size << std::endl;
    throw std::invalid_argument("input_dataのサイズとshapeの総要素数が一致しません。");
  }

  NTensor<rank> tensor(shape);

  // データをテンソルにコピー
  Eigen::Index idx = 0;
  for (auto it = input_data.begin(); it != input_data.end(); ++it, ++idx)
  {
    tensor.data()[idx] = *it;
  }

  return tensor;
}

template<int rank>
NTensor<rank> StepFunction(const NTensor<rank> &x)
{
  return x.unaryExpr([](Float x)
                     { return x > 0 ? 1.f : 0.f; });
}

template<int rank>
NTensor<rank> Sigmoid(const NTensor<rank> &x)
{
  return x.unaryExpr([](Float x)
                     { return 1 / (1 + std::exp(-x)); });
}

template<int rank>
NTensor<rank> ReLU(const NTensor<rank> &x)
{
  return x.unaryExpr([](Float x)
                     { return std::max(0.f, x); });
}

/**
 * テンソル間の行列積を計算する関数
 * バッチ次元を考慮し、最後の2次元を行列として扱う
 * @tparam RankA テンソルAのランク
 * @tparam RankB テンソルBのランク
 * @param A テンソルA（形状：[batch_dims..., M, K]）
 * @param B テンソルB（形状：[batch_dims..., K, N]）
 * @return 行列積の結果（形状：[batch_dims..., M, N]）
 */
template<int RankA, int RankB>
auto Dot(const NTensor<RankA> &A, const NTensor<RankB> &B)
{
  return A.contract(B, Eigen::array<Eigen::IndexPair<int>, 1>{
    Eigen::IndexPair<int>(RankA - 1, RankB - 2)});
}

template<int rank>
NTensor<rank> operator+(const NTensor<rank> &lhs, const NTensor<rank> &rhs)
{
  return lhs + rhs;
}

template<int rank>
NTensor<rank> operator-(const NTensor<rank> &lhs, const NTensor<rank> &rhs)
{
  return lhs - rhs;
}

template<int rank>
NTensor<rank> operator*(const NTensor<rank> &lhs, const NTensor<rank> &rhs)
{
  return lhs * rhs;
}

template<int rank>
NTensor<rank> operator/(const NTensor<rank> &lhs, const NTensor<rank> &rhs)
{
  return lhs / rhs;
}

/**
 * Softmax関数
 * 入力テンソルの最後の次元に沿ってソフトマックスを計算します。
 * @tparam rank テンソルの次元数
 * @param input_tensor 入力テンソル
 * @return ソフトマックスを適用したテンソル
 */
template<int rank>
NTensor<rank> Softmax(const NTensor<rank> &input_tensor, const float beta = 1.0f)
{
  // ソフトマックスを計算する次元は最後の次元（rank - 1）
  constexpr int softmax_dim = rank - 1;
  std::cout << "softmax_dim : " << softmax_dim << std::endl;



  // 合計を計算
  if constexpr (softmax_dim == 0)
  {
    // TODO : 最大値を引くことでオーバーフローを防ぐ処理を追加

    // expを計算
    auto exp_tensor = input_tensor.exp();
    std::cout << "exp_tensor : " << exp_tensor << std::endl;

    std::cout << "dim 0" << std::endl;

    auto sum_tensor = exp_tensor.sum();
    std::cout << "sum_tensor : " << sum_tensor << std::endl;

    // reshape で次元を合わせる
    auto beta_tensor = NTensor<rank>(input_tensor.dimensions());
    beta_tensor.setConstant(beta);

    NTensor<rank> result;
    result = exp_tensor / beta_tensor;

    return result;
  }
  else
  {
    // expを計算
    auto exp_tensor = input_tensor.exp();
    std::cout << "exp_tensor : " << exp_tensor << std::endl;
    std::cout << "dim 1" << std::endl;

    auto sum_tensor = exp_tensor.sum(softmax_dim);
    std::cout << "sum_tensor : " << sum_tensor << std::endl;

    NTensor<rank> result;


    return result;
  }

}

template<int rank>
auto CrossEntropyError(const NTensor<rank> &y, const NTensor<rank> &t)
{
  std::cout << "y : " << y << std::endl;
  std::cout << "t : " << t << std::endl;
  constexpr Float delta = 1e-7;

  constexpr int target_rank = rank - 1;
  if constexpr (target_rank == 0)
  {
    std::cout << "rank 0" << std::endl;
    return -1 * (t * (y + delta).log()).sum();
  }
  else
  {
    std::cout << "rank 1" << std::endl;
    return -1 * (t * (y + delta).log()).sum(target_rank);
  }
}

} // namespace nagato::na

#endif //DEZERO1__NTENSOR_HPP_
