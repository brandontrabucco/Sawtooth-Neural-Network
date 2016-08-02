/*
 * SerriformNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "SerriformNetwork.h"

SerriformNetwork::SerriformNetwork(int is, int o, double l, double d) {
	// TODO Auto-generated constructor stub
	inputSize = is;
	overlap = o;
	learningRate = l;
	decayRate = d;
}

SerriformNetwork::~SerriformNetwork() {
	// TODO Auto-generated destructor stub
}

int SerriformNetwork::getPreviousNeurons() {
	int sum = 0;
	for (int i = ((int)layers.size() - 1); i >= ((int)layers.size() - overlap - 2); i--) {
		if (i == -1) sum += inputSize;
		else if (i >= 0) {
			sum += (int)layers[i].size();
		}
	}
	return sum;
}

void SerriformNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	vector<double> e;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
		e.push_back(0.0);
	} layers.push_back(buffer);
	error.push_back(e);
}

vector<double> SerriformNetwork::classify(vector<double> input) {
	vector<double> output(layers[layers.size() - 1].size());
	if ((int)input.size() == inputSize) {
		// calculate activations in reverse order from top
		for (int i = (layers.size() - 1); i >= 0; i--) {
#pragma omp parallel for
			for (int j = 0; j < (int)layers[i].size(); j++) {
				// sum the input from all previous layer neurons
				vector<double> connections;
				for (int k = (i - overlap - 1); k < i; k++) {
					if (k == -1) connections = input;
					else if (k >= 0) for (int l = 0; l < (int)layers[k].size(); l++) {
						connections.push_back(layers[k][l].activation);
					}
				}
				// compute the activation
				double result = layers[i][j].forward(connections);
				// if at top of network, push to output
				if (i == ((int)layers.size() - 1)) output[j] = (result);
			}
		}
		return output;
	} else return output;
}

vector<double> SerriformNetwork::train(vector<double> input, vector<double> target) {
	if ((int)input.size() == inputSize && (int)target.size() == ((int)layers[layers.size() - 1].size())) {
		// calculate activations in reverse order from top
		for (int i = ((int)layers.size() - 1); i >= 0; i--) {
#pragma omp parallel for
			for (int j = 0; j < (int)layers[i].size(); j++) {
				// sum the input from all previous layer neurons
				vector<double> connections;
				for (int k = (i - overlap - 1); k < i; k++) {
					if (k == -1) connections = input;
					else if (k >= 0) for (int l = 0; l < (int)layers[k].size(); l++) {
						connections.push_back(layers[k][l].activation);
					}
				}
				// compute the activation
				double result = layers[i][j].forward(connections);
				// initialize error at top of network
				if (i == ((int)layers.size() - 1)) error[i][j] = (result - target[j]);
				// propogate the error back through node [i][j]
				vector<double> temp = (layers[i][j].backward(error[i][j], learningRate));	// error is uninitialized
				// sum the weighted error for all previous nodes
				int offset = 0;
#pragma omp critical
				for (int k = (i - overlap - 1); k < i; k++) {
					if (k == -1) offset += inputSize;
					else if (k >= 0) for (int l = 0; l < (int)layers[k].size(); l++) {
						offset++;
						error[k][l] += temp[offset];
					}
				}
			}
		} learningRate *= decayRate;
		return error[layers.size() - 1];
	}
	else return vector<double>(0);
}
