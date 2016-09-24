/*
 * Neuron.cpp
 *
 *  Created on: Jun 22, 2016
 *      Author: trabucco
 */

#include "Neuron.h"

long long Neuron::n = 0;

Neuron::Neuron(int connections) {
	// TODO Auto-generated constructor stub
	activation.push_back(0);
	activationPrime.push_back(0);
	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);
	for (int i = 0; i < connections; i++) {
		weight.push_back(d(g));
	}
}

Neuron::~Neuron() {
	// TODO Auto-generated destructor stub
}

double Neuron::sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

double Neuron::sigmoidPrime(double input) {
	return sigmoid(input) * (1 - sigmoid(input));
}

double Neuron::activate(double input) {
	return tanh(input);
}

double Neuron::activatePrime(double input) {
	return (1 - (tanh(input) * tanh(input)));
}

double Neuron::forward(vector<double> input) {
	double sum = 0;
	impulse.push_back(input);

	// find the weighted sum of all input
	for (int i = 0; i < input.size(); i++) {
		sum += input[i] * weight[i];
	} double a = activate(sum);
	activation.push_back(a);
	activationPrime.push_back(activatePrime(sum));
	return a;
}

vector<double> Neuron::backward(double errorPrime, double learningRate, int t, int length) {
	vector<double> weightedError;
	// update all weights
	for (int i = 0; i < weight.size(); i++) {
		weightedError.push_back(errorPrime * weight[i] * activationPrime[t]);
		weight[i] -= (learningRate * errorPrime * activationPrime[t] * impulse[t][i]) / length;
	} return weightedError;
}

void Neuron::clear() {
	impulse.clear();
	activation.clear();
	activationPrime.clear();
}
