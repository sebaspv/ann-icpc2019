#include <bits/stdc++.h>
using namespace std;

int sign(int x) {
    if (x == 0) return -1;
    else return x / abs(x); // Ensures sign is -1 or 1
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
    if (outputSign == 1) return 1;
    else return -1;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidPD(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

int discreteSigmoid(int weight) {
    double sigmoidPDOutput = sigmoidPD((double)weight);
    return (sigmoidPDOutput < 0.5) ? 0 : 1;
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

    void backpropagation(int predicted, int actual) {
        int error = actual - predicted;
        if (error != 0) {
            for (auto& perceptron : perceptronsClass0) {
                for (auto& weight : perceptron.inputs) {
                    weight = (weight == 1) ? -1 : 1; // Flip between -1 and 1
                }
            }
            for (auto& perceptron : perceptronsClass1) {
                for (auto& weight : perceptron.inputs) {
                    weight = (weight == 1) ? -1 : 1; // Flip between -1 and 1
                }
            }
        }
    }

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

    NeuralNet neuralNet(inputBytes[0]);

    neuralNet.train(inputBytes, labels, 10); // Train for 10 epochs
    for (auto weights0: neuralNet.perceptronsClass0){
      for (auto weight: weights0.inputs){
        cout << weight << " ";
      }
      cout << "\n";
    }
    for (auto weights1: neuralNet.perceptronsClass1){
      for (auto weight: weights1.inputs){
        cout << weight << " ";
      }
      cout << "\n";
    }
    return 0;
}
