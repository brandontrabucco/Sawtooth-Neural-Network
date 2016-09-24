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
	timestep = -1;
}

SerriformNetwork::~SerriformNetwork() {
	// TODO Auto-generated destructor stub
}

int SerriformNetwork::getPreviousNeurons() {
	int sum = inputSize;
	for (unsigned int i = 0; i < layers.size(); i++)
		sum += layers[i].size();
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

vector<double> SerriformNetwork::forward(vector<double> input) {
	vector<double> output;
	if (input.size() == inputSize) {
		// calculate activations in reverse order from top
		errorBuffer.push_back(error);
		timestep++;
		for (int i = (layers.size() - 1); i >= 0; i--) {
			for (int j = 0; j < layers[i].size(); j++) {
				// sum the input from all previous  neurons
				vector<double> connections = input;
				for (int k = 0; k < i; k++) {
					for (int l = 0; l < layers[k].size(); l++) {
						connections.push_back(layers[k][l].activation[timestep]);
					}
				}
				// compute the activation
				double result = layers[i][j].forward(connections);

				// initialize error at top of network
				if (i == (layers.size() - 1)) output.push_back(result);
				// if at top of network, push to output
			}
		} return output;
	}  else return output;
}

vector<double> SerriformNetwork::forward(vector<double> input, vector<double> target) {
	if (input.size() == inputSize) {
		// calculate activations in reverse order from top
		errorBuffer.push_back(error);
		timestep++;
		for (int i = (layers.size() - 1); i >= 0; i--) {
			for (int j = 0; j < layers[i].size(); j++) {
				// sum the input from all previous  neurons
				vector<double> connections = input;
				for (int k = 0; k < i; k++) {
					for (int l = 0; l < layers[k].size(); l++) {
						connections.push_back(layers[k][l].activation[timestep]);
					}
				}
				// compute the activation
				double result = layers[i][j].forward(connections);

				// initialize error at top of network
				if (i == (layers.size() - 1)) errorBuffer[timestep][i][j] = (result - target[j]);
				// if at top of network, push to output
			}
		} return errorBuffer[timestep][layers.size() - 1];
	} else return vector<double>();
}

void SerriformNetwork::backward() {
	// calculate activations in reverse order from top
	for (; timestep >= 0; timestep--) {
		for (int i = (layers.size() - 1); i >= 0; i--) {
			for (int j = 0; j < layers[i].size(); j++) {
				// propogate the error back through time
				//cout << endl << "Backward pass " << t << " layer " << i << " " << j << " error: " << errorBuffer[t][i][j] << endl;
				vector<double> temp = (layers[i][j].backward(errorBuffer[timestep][i][j], learningRate, timestep, errorBuffer.size()));
				// sum the weighted error for all previous nodes
				int offset = inputSize;
				if ((timestep > 0)) for (int k = 0; k < i; k++) {	// previous layers
					for (int l = 0; l < layers[k].size(); l++) {	// previous neurons
						errorBuffer[timestep - 1][k][l] += temp[offset];
						offset++;
					}
				}
			}
		} learningRate *= decayRate;
	}
}

void SerriformNetwork::clear() {
	for (int i = (layers.size() - 1); i >= 0; i--) {
		for (int j = 0; j < layers[i].size(); j++) {
			layers[i][j].clear();
		}
	}
	errorBuffer.clear();
	timestep = -1;
}
