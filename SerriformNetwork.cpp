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
	maxLayerSize = 0;
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
	if (size > maxLayerSize) maxLayerSize = size;
}

vector<double> SerriformNetwork::forward(vector<double> input) {
	vector<double> output(layers[layers.size() - 1].size());
	if (input.size() == inputSize) {
		// calculate activations in reverse order from top
		errorBuffer.push_back(error);
		timestep++;
		/**
		 *
		 * This is a parallel loop
		 *
		 */
		#pragma omp parallel for schedule(dynamic, 1) collapse(2)
		for (int i = (layers.size() - 1); i >= 0; i--) {
			for (int j = 0; j < maxLayerSize; j++) {
				if (j < layers[i].size()) {
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
					if (i == (layers.size() - 1)) output[j] = result;
					// if at top of network, push to output
				}
			}
		} return output;
	}  else return output;
}

vector<double> SerriformNetwork::forward(vector<double> input, vector<double> target) {
	if (input.size() == inputSize) {
		// calculate activations in reverse order from top
		errorBuffer.push_back(error);
		timestep++;
		/**
		 *
		 * This is a parallel loop
		 *
		 */
		#pragma omp parallel for schedule(dynamic, 1) collapse(2)
		for (int i = (layers.size() - 1); i >= 0; i--) {
			for (int j = 0; j < maxLayerSize; j++) {
				if (j < layers[i].size()) {
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
			}
		} return errorBuffer[timestep][layers.size() - 1];
	} else return vector<double>();
}

void SerriformNetwork::backward() {
	// calculate activations in reverse order from top
	for (; timestep >= 0; timestep--) {
		/**
		 *
		 * This is a parallel loop
		 *
		 */
		#pragma omp parallel for schedule(dynamic, 1) collapse(2)
		for (int i = (layers.size() - 1); i >= 0; i--) {
			for (int j = 0; j < maxLayerSize; j++) {
				if (j < layers[i].size()) {
					// propogate the error back through time
					vector<double> temp = (layers[i][j].backward(errorBuffer[timestep][i][j], learningRate, timestep, errorBuffer.size()));

					// sum the weighted error for all previous nodes
					int offset = inputSize;
					if ((timestep > 0)) for (int k = 0; k < i; k++) {	// previous layers
						for (int l = 0; l < layers[k].size(); l++) {	// previous neurons
							#pragma omp atomic
							errorBuffer[timestep - 1][k][l] += temp[offset];
							offset++;
						}
					} else {
						layers[i][j].update();
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

void SerriformNetwork::saveToFile(string fileName) {
	ofstream learningDataFile(fileName + ".brain");
	if (!learningDataFile.is_open()) return;

	for (int i = 0; i < layers.size(); i++) {
		for (int j = 0; j < layers[i].size(); j++) {
			for (int k = 0; k < layers[i][j].weight.size(); k++) {
				if (k != 0) learningDataFile << " ";
				learningDataFile << layers[i][j].weight[k];
			} learningDataFile << endl;
		} learningDataFile << endl;
	}
}
