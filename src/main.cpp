#include <bits/stdc++.h>

using namespace std;

int sign(int x){
  if (x == 0) return -1;
  else return x/x;
}

int argmax(vector<int> weights){
  int maxso = INT_MIN;
  int outputSign = 1;
  for (const auto &weight: weights){
    if (abs(weight) >= maxso){
      maxso = weight;
      outputSign = sign(weight);
    }
  }
  if (outputSign == 1){
    return 1;
  }
  return 0;
}

int inputSign(vector<int> &weights, vector<int> &input){
  int accum = 0;
  for (int inputInd = 0; inputInd < (int)weights.size(); inputInd++){
    accum+=(weights[inputInd] * input[inputInd]);
  }
  return sign(accum);
}

struct ArgMaxNode {
  vector<int> weights;
  int output = argmax(weights);
};

struct SumNode {
  vector<int> weights;
  int output = accumulate(weights.begin(), weights.end(), 0);
};

struct PerceptronNode {
  vector<int> weights; // Always size 51
  int sign;
};

int main(){

  return 0;
}
