
#include "NTensor.hpp"

struct Network
{
  nagato::na::NTensor<2> W1;
  nagato::na::NTensor<1> B1;
  nagato::na::NTensor<2> W2;
  nagato::na::NTensor<1> B2;
  nagato::na::NTensor<2> W3;
  nagato::na::NTensor<1> B3;
};

auto InitNetwork()
{
  Network network;

  network.W1 = nagato::na::CreateTensor<2>(
    {
      {0.1, 0.3, 0.5},
      {0.2, 0.4, 0.6}
    },
    {2, 3}
  );
  network.B1 = nagato::na::CreateTensor<1>({0.1, 0.2, 0.3}, {3});
  network.W2 = nagato::na::CreateTensor<2>(
    {
      {0.1, 0.4},
      {0.2, 0.5},
      {0.3, 0.6}
    },
    {3, 2}
  );
  network.B2 = nagato::na::CreateTensor<1>({0.1, 0.2}, {2});
  network.W3 = nagato::na::CreateTensor<2>(
    {
      {0.1, 0.3},
      {0.2, 0.4}
    },
    {2, 2}
  );
  network.B3 = nagato::na::CreateTensor<1>({0.1, 0.2}, {2});

  return network;
}

auto Forward(const Network &network, const nagato::na::NTensor<1> &x)
{
  using namespace nagato;
  na::NTensor<1> a1 = x * network.W1 + network.B1;
  na::NTensor<1> z1 = nagato::na::Sigmoid(a1);
  na::NTensor<1> a2 = z1 * network.W2 + network.B2;
  na::NTensor<1> z2 = nagato::na::Sigmoid(a2);
  na::NTensor<1> a3 = z2 * network.W3 + network.B3;
  na::NTensor<1> y = a3;

  return y;
}

int main()
{
  using namespace nagato;

  auto t = na::CreateTensor<1>({0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, {10});
  auto y = na::CreateTensor<1>({0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0}, {10});

  auto loss = na::CrossEntropyError(y, t);

  std::cout << "loss : " << loss << std::endl;
  return 0;
}