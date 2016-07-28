/*
 * SawtoothNetwork.cpp
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#include "SawtoothNetwork.h"

SawtoothNetwork::SawtoothNetwork(int is) {
	// TODO Auto-generated constructor stub
	inputSize = is;
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
	for (int i = 0; i < size; i++)
		buffer.push_back(Neuron(getPreviousNeurons()));
	layers.push_back(buffer);
}

vector<double> SawtoothNetwork::feedforward(vector<double> input) {
	vector<double> output;
	if (input.size() == inputSize) {
		// calculate activations in reverse order from top
		for (int i = (layers.size() - 1); i >= 0; i--) {
			for (int j = 0; j < layers[i].size(); j++) {
				// sum the input from all previous layer neurons
				vector<double> connections = input;
				for (int k = 0; k < (i - 1); k++)
					for (int l = 0; l < layers[i].size(); l++)
						connections.push_back(layers[k][l].activation);
				double result = layers[i][j].forward(connections);
				if (i == (layers.size() - 1)) output.push_back(result);
			}
		}
		return output;
	} else return output;
}
