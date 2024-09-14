#include <bits/stdc++.h>
using namespace std;

int sign(int x) {
    return (x >= 0) ? 1 : -1;
}

int argmax(vector<int> weights) {
    int maxso = INT_MIN;
    int outputSign = -1;
    for (const auto& weight : weights) {
        if (abs(weight) >= maxso) {
            maxso = weight;
            outputSign = sign(weight);
        }
    }
    return outputSign;
}

// Sigmoid derivative function for discrete values
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidPD(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

// Discrete sigmoid derivative based on the discrete output of weights
int DiscreteSigmoid(int weight) {
    double SigmoidPDOutput = sigmoidPD((double)weight);
    return (SigmoidPDOutput < 0.5) ? 0 : 1;
}

int inputSign(const vector<int>& weights, const vector<int>& input) {
    int accum = 0;
    for (int inputInd = 0; inputInd < (int)weights.size(); inputInd++) {
        accum += (weights[inputInd] * input[inputInd]);
    }
    return sign(accum);
}

class ArgmaxNode {
private:
    vector<int> weights;
    int output;
public:
    ArgmaxNode() {
        this->output = -1;
    }
    ArgmaxNode(vector<int> weightInputs) {
        this->weights = weightInputs;
        this->output = argmax(weights);
    }
    int Calculate(const vector<int>& weightInputs) {
        this->weights = weightInputs;
        this->output = argmax(weights);
        return this->output;
    }
};

struct PerceptronNode {
    vector<int> inputs; // Size 51
    int sign;
};

int accumulatePerceptrons(const vector<PerceptronNode>& weights) {
    int output = 0;
    for (const auto& perceptron : weights) {
        output += perceptron.sign;
    }
    return output;
}

class SumNode {
private:
    int output;
    vector<PerceptronNode> inputs;
public:
    SumNode() : output(0) {}
    int Calculate(const vector<PerceptronNode>& newInputs) {
        this->inputs = newInputs;
        this->output = accumulatePerceptrons(this->inputs);
        return this->output;
    }
};

class NeuralNet {
public:
    vector<int> inputs;
    vector<PerceptronNode> perceptronsClass0;
    vector<PerceptronNode> perceptronsClass1;
    SumNode class0;
    SumNode class1;
    ArgmaxNode output;
    
    NeuralNet(vector<int> inputs) {
        this->inputs = inputs;
        // Initialize perceptrons for both classes with random weights (-1 or 1)
        for (int perceptronInd = 0; perceptronInd < 15; perceptronInd++) {
            PerceptronNode perceptron;
            for (int i = 0; i < 51; i++) {
                perceptron.inputs.push_back((rand() % 2 == 0) ? -1 : 1); // Random -1 or 1
            }
            perceptron.sign = sign(accumulate(perceptron.inputs.begin(), perceptron.inputs.end(), 0));
            perceptronsClass0.push_back(perceptron);
        }
        for (int perceptronInd = 0; perceptronInd < 15; perceptronInd++) {
            PerceptronNode perceptron;
            for (int i = 0; i < 51; i++) {
                perceptron.inputs.push_back((rand() % 2 == 0) ? -1 : 1);
            }
            perceptron.sign = sign(accumulate(perceptron.inputs.begin(), perceptron.inputs.end(), 0));
            perceptronsClass1.push_back(perceptron);
        }
    }

    // Forward pass through the network
    int forward() {
        for (int i = 0; i < (int)perceptronsClass0.size(); i++) {
            int dot_product = 0;
            for (int j = 0; j < (int)inputs.size(); j++) {
                dot_product += inputs[j] * perceptronsClass0[i].inputs[j];
            }
            perceptronsClass0[i].sign = sign(dot_product);
        }
        for (int i = 0; i < (int)perceptronsClass1.size(); i++) {
            int dot_product = 0;
            for (int j = 0; j < (int)inputs.size(); j++) {
                dot_product += inputs[j] * perceptronsClass1[i].inputs[j];
            }
            perceptronsClass1[i].sign = sign(dot_product);
        }

        int sumClass0 = class0.Calculate(perceptronsClass0);
        int sumClass1 = class1.Calculate(perceptronsClass1);

        vector<int> sumOutputs = { sumClass0, sumClass1 };
        return output.Calculate(sumOutputs);
    }

    // Backpropagation using the discrete sigmoid
    void backpropagation(int predicted, int actual) {
        int error = actual - predicted;

        if (error != 0) {
            // Update the weights using the DiscreteSigmoid function
            for (auto& perceptron : perceptronsClass0) {
                for (auto& weight : perceptron.inputs) {
                    // Calculate the gradient using the discrete sigmoid function
                    int sigmoidDerivative = DiscreteSigmoid(weight);

                    // Update the weight based on the gradient
                    if (sigmoidDerivative > 0) {
                        weight += error * sigmoidDerivative;
                    } else {
                        weight -= error * sigmoidDerivative;
                    }

                    // Ensure the weight remains -1 or 1
                    weight = (weight > 0) ? 1 : -1;
                }
            }

            for (auto& perceptron : perceptronsClass1) {
                for (auto& weight : perceptron.inputs) {
                    // Calculate the gradient using the discrete sigmoid function
                    int sigmoidDerivative = DiscreteSigmoid(weight);

                    // Update the weight based on the gradient
                    if (sigmoidDerivative > 0) {
                        weight += error * sigmoidDerivative;
                    } else {
                        weight -= error * sigmoidDerivative;
                    }

                    // Ensure the weight remains -1 or 1
                    weight = (weight > 0) ? 1 : -1;
                }
            }
        }
    }

    // Forward pass on the entire dataset and backpropagation
    void train(vector<vector<int>>& dataset, vector<int>& labels, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < (int)dataset.size(); i++) {
                this->inputs = dataset[i];
                int predicted = forward();
                int actual = labels[i];
                backpropagation(predicted, actual);
            }
        }
    }
};

const int INPUT_SIZE = 11546;

int main() {
    vector<vector<int>> inputBytes; // X
    vector<int> labels; // Y
    for (int i = 0; i < INPUT_SIZE; i++) {
        vector<int> inputB(51);
        int label;
        for (int j = 0; j < 51; j++) {
            cin >> inputB[j];
        }
        cin >> label;
        inputBytes.push_back(inputB);
        labels.push_back(label);
    }

    NeuralNet neuralNet(inputBytes[0]); // Initialize with first input

    neuralNet.train(inputBytes, labels, 2);
    for (auto aa: neuralNet.perceptronsClass0){
      for (auto b: aa.inputs){
        cout << b << " ";
      }
      cout << "\n";
    }
    for (auto aa: neuralNet.perceptronsClass1){
      for (auto b: aa.inputs){
        cout << b << " ";
      }
      cout << "\n";
    }
    return 0;
}
