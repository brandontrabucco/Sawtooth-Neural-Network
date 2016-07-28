/*
 * SawtoothNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "SawtoothNetwork.h"

SawtoothNetwork::SawtoothNetwork(int is, double l, double d) {
	// TODO Auto-generated constructor stub
	inputSize = is;
	learningRate = l;
	decayRate = d;
}

SawtoothNetwork::~SawtoothNetwork() {
	// TODO Auto-generated destructor stub
}

int SawtoothNetwork::getPreviousNeurons() {
	int sum = inputSize;
	for (int i = 0; i < layers.size(); i++)
		sum += layers[i].size();
	return sum;
}

void SawtoothNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	vector<double> e;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
		e.push_back(0.0);
	} layers.push_back(buffer);
	error.push_back(e);
}

vector<double> SawtoothNetwork::classify(vector<double> input) {
	vector<double> output;
	if (input.size() == inputSize) {
		// calculate activations in reverse order from top
		for (int i = (layers.size() - 1); i >= 0; i--) {
			for (int j = 0; j < layers[i].size(); j++) {
				// sum the input from all previous layer neurons
				vector<double> connections = input;
				for (int k = 0; k < i; k++)
					for (int l = 0; l < layers[i].size(); l++)
						connections.push_back(layers[k][l].activation);
				// compute the activation
				double result = layers[i][j].forward(connections);
				// if at top of network, push to output
				if (i == (layers.size() - 1)) output.push_back(result);
			}
		}
		return output;
	} else return output;
}

vector<double> SawtoothNetwork::train(vector<double> input, vector<double> target) {
	vector<double> output;
	if (input.size() == inputSize && target.size() == (layers[layers.size() - 1].size())) {
		// calculate activations in reverse order from top
		for (int i = (layers.size() - 1); i >= 0; i--) {
			for (int j = 0; j < layers[i].size(); j++) {
				// sum the input from all previous layer neurons
				vector<double> connections = input;
				for (int k = 0; k < i; k++)
					for (int l = 0; l < layers[i].size(); l++)
						connections.push_back(layers[k][l].activation);
				// compute the activation
				double result = layers[i][j].forward(connections);
				// initialize error at top of network
				if (i == (layers.size() - 1)) error[i][j] = (result - target[j]);
				// propogate the error back through node [i][j]
				vector<double> temp = (layers[i][j].backward(error[i][j], learningRate));
				// sum the weighted error for all previous nodes
				for (int k = 0; k < i; k++)
					for (int l = 0; l < layers[i].size(); l++)
						error[k][l] += temp[k * (layers[i].size()) + l];
			}
		}


		return output;
	}
	else return output;
}
