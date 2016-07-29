/*
 * SerriformNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "SerriformNetwork.h"

SerriformNetwork::SerriformNetwork(int is, double l, double d) {
	// TODO Auto-generated constructor stub
	inputSize = is;
	learningRate = l;
	decayRate = d;
}

SerriformNetwork::~SerriformNetwork() {
	// TODO Auto-generated destructor stub
}

int SerriformNetwork::getPreviousNeurons() {
	int sum = inputSize;
	for (unsigned int i = 0; i < blocks.size(); i++)
		sum += blocks[i].size();
	return sum;
}

void SerriformNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	vector<double> e;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
		e.push_back(0.0);
	} blocks.push_back(buffer);
	error.push_back(e);
}

vector<double> SerriformNetwork::classify(vector<double> input) {
	vector<double> output;
	if (input.size() == inputSize) {
		// calculate activations in reverse order from top
		for (int i = (blocks.size() - 1); i >= 0; i--) {
			for (int j = 0; j < blocks[i].size(); j++) {
				// sum the input from all previous layer neurons
				vector<double> connections = input;
				for (int k = 0; k < i; k++)
					for (int l = 0; l < blocks[k].size(); l++)
						connections.push_back(blocks[k][l].activation);
				// compute the activation
				double result = blocks[i][j].forward(connections);
				// if at top of network, push to output
				if (i == (blocks.size() - 1)) output.push_back(result);
			}
		}
		return output;
	} else return output;
}

vector<double> SerriformNetwork::train(vector<double> input, vector<double> target) {
	if (input.size() == inputSize && target.size() == (blocks[blocks.size() - 1].size())) {
		// calculate activations in reverse order from top
		for (int i = (blocks.size() - 1); i >= 0; i--) {
			for (int j = 0; j < blocks[i].size(); j++) {	// error is here
				// sum the input from all previous layer neurons
				vector<double> connections = input;
				for (int k = 0; k < i; k++)
					for (int l = 0; l < blocks[k].size(); l++)
						connections.push_back(blocks[k][l].activation);
				// compute the activation
				double result = blocks[i][j].forward(connections);
				// initialize error at top of network
				if (i == (blocks.size() - 1)) error[i][j] = (result - target[j]);
				// propogate the error back through node [i][j]
				vector<double> temp = (blocks[i][j].backward(error[i][j], learningRate));
				// sum the weighted error for all previous nodes
				for (int k = 0; k < i; k++)
					for (int l = 0; l < blocks[k].size(); l++)
						error[k][l] += temp[k * (blocks[i].size()) + l];
			}
		} learningRate *= decayRate;
		return error[blocks.size() - 1];
	}
	else return vector<double>(0);
}
