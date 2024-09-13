#include <bits/stdc++.h>

using namespace std;

int sign(int x){
  if (x == 0) return -1;
  else return x/x;
}

int argmax(vector<int> weights){
  int maxso = INT_MIN;
  int outputSign = -1;
  for (const auto &weight: weights){
    if (abs(weight) >= maxso){
      maxso = weight;
      outputSign = sign(weight);
    }
  }
  if (outputSign == 1){
    return 1;
  }
  else if (outputSign == -1){
    return -1;
  }
  return 0;
}

double Sigmoid(int weight){
  double ePow = exp((double)weight);
  return (ePow)/((double)1 + ePow);
}

double SigmoidPD(double x){
  return Sigmoid(x)*(1-Sigmoid(x));
}

int DiscreteSigmoid(int weight){
  double SigmoidPDOutput = SigmoidPD((double)weight);
  if (SigmoidPDOutput < 0.5){
    return 0;
  }
  return 1;
}

int inputSign(vector<int> &weights, vector<int> &input){
  int accum = 0;
  for (int inputInd = 0; inputInd < (int)weights.size(); inputInd++){
    accum+=(weights[inputInd] * input[inputInd]);
  }
  return sign(accum);
}

class ArgmaxNode {
  private:
  vector<int> weights;
  int output;
  public:
  ArgmaxNode(vector<int> weightInputs){
    this->weights = weightInputs;
    this->output = argmax(weights);
  }
  int Calculate(vector<int> weightInputs){
    this->weights = weightInputs;
    this->output = argmax(weights);
    return this->output;
  }
};

struct PerceptronNode {
  vector<int> inputs; // Always size 51
  int sign;
};

int accumulatePerceptrons(vector<PerceptronNode> weights){
  int output = 0;
  for (const auto& Perceptron : weights){
    output+=Perceptron.sign;
  }
  return output;
}

class SumNode {
  private:
  int output;
  vector<PerceptronNode> inputs;
  public:
  SumNode(){
    output = accumulatePerceptrons(inputs);
  }
  int Calculate(vector<PerceptronNode> newInputs){
    this->inputs = newInputs;
    this->output = accumulatePerceptrons(this->inputs);
    return this->output;
  }
};

struct NeuralNetwork {
  vector<int> inputs;
  vector<PerceptronNode> perceptronsClass0;
  vector<PerceptronNode> perceptronClass1;
  SumNode class0;
  SumNode class1;
  ArgmaxNode output;
};

const int INPUT_SIZE = 11546;

int main(){
  vector<vector<int>> inputBytes;
  vector<int> labels;
  for (int i = 0; i < INPUT_SIZE; i++){
    vector<int> inputB(51);
    int label;
    for (int j = 0; j < 51; j++){
      cin >> inputB[j];
    }
    cin >> label;
    inputBytes.push_back(inputB);
    labels.push_back(label);
  }
  return 0;
}
