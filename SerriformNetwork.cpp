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
	for (int i = 0; i < layers.size(); i++) {
		free(error[i]);
	} free(error);
}

int SerriformNetwork::getPreviousNeurons() {
	int sum = 0;
	for (int i = ((int)layers.size() - 1); i > ((int)layers.size() - overlap - 2); i--) {
		if (i == -1) sum += inputSize;
		else if (i >= 0) {
			sum += (int)layers[i].size();
		}
	}
	return sum;
}

int SerriformNetwork::getPreviousNeurons(int l) {
	int sum = 0;
	for (int i = (l - 1); i > (l - overlap - 2); i--) {
		if (i == -1) sum += inputSize;
		else if (i >= 0) {
			sum += (int)layers[i].size();
		} //cout << i << " " << sum << endl;
	}
	return sum;
}

void SerriformNetwork::addLayer(int size) {
	vector<Neuron> buffer;
	for (int i = 0; i < size; i++) {
		buffer.push_back(Neuron(getPreviousNeurons()));
	} layers.push_back(buffer);
	if (layers.size() > 1) error = (double **)realloc(error, (sizeof(double *) * layers.size()));
	else error = (double **)malloc(sizeof(double *) * layers.size());
	error[layers.size() - 1] = (double *)calloc(size, sizeof(double));
}

vector<double> SerriformNetwork::classify(vector<double> input) {
	double *output = (double *)malloc(sizeof(double) * layers[layers.size() - 1].size());
	// calculate activations in reverse order from top
	for (int i = (layers.size() - 1); i >= 0; i--) {
		#pragma omp parallel for
		for (int j = 0; j < (int)layers[i].size(); j++) {
			// sum the input from all previous layer neurons
			double *connections = (double *)malloc(sizeof(double) * getPreviousNeurons(i));	// faulty alloc
			int offset = 0;
			for (int k = (i - overlap - 1); k < i; k++) {
				if (k == -1) {
					copy(input.begin(), input.end(), connections);
					offset += inputSize;
				} else if (k >= 0) for (int l = 0; l < (int)layers[k].size(); l++) {
					connections[offset] = (layers[k][l].activation);	// illegal write
					offset++;
				}
			}
			// compute the activation
			double activation = layers[i][j].forward(connections);
			free(connections);
			// if at top of network, push to output
			if (i == ((int)layers.size() - 1)) output[j] = (activation);
		}
	} vector<double> result(&output[0], &output[layers[layers.size() - 1].size()]);
	free(output);
	return result;
}

vector<double> SerriformNetwork::train(vector<double> input, vector<double> target) {
	if ((int)input.size() == inputSize && (int)target.size() == ((int)layers[layers.size() - 1].size())) {
		// calculate activations in reverse order from top
		for (int i = ((int)layers.size() - 1); i >= 0; i--) {
			#pragma omp parallel for
			for (int j = 0; j < (int)layers[i].size(); j++) {
				// sum the input from all previous layer neurons
				double *connections = (double *)malloc(sizeof(double) * getPreviousNeurons(i));	// faulty alloc
				int offset = 0;
				for (int k = (i - overlap - 1); k < i; k++) {
					if (k == -1) {
						copy(input.begin(), input.end(), connections);
						offset += inputSize;
					} else if (k >= 0) for (int l = 0; l < (int)layers[k].size(); l++) {
						connections[offset] = (layers[k][l].activation);	// illegal write
						offset++;
					}
				}
				// compute the activation
				double activation = layers[i][j].forward(connections);
				free(connections);
				// initialize error at top of network
				if (i == ((int)layers.size() - 1)) error[i][j] = (activation - target[j]);
				// propogate the error back through node [i][j]
				double *temp = (layers[i][j].backward(error[i][j], learningRate));
				// sum the weighted error for all previous nodes
				offset = 0;
				#pragma omp critical
				for (int k = (i - overlap - 1); k < i; k++) {
					if (k == -1) offset += inputSize;
					else if (k >= 0) for (int l = 0; l < (int)layers[k].size(); l++) {
						error[k][l] += temp[offset];
						offset++;
					}
				} free(temp);
			}
		} vector<double> result(&error[layers.size() - 1][0], &error[layers.size() - 1][layers[layers.size() - 1].size()]);
		learningRate *= decayRate;
		return result;
	}
	else return vector<double>();
}
