/*
 * SawtoothNetwork.h
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#ifndef SERRIFORMNETWORK_H_
#define SERRIFORMNETWORK_H_

#include <vector>
#include "Neuron.h"
using namespace std;

class SerriformNetwork {
private:
	unsigned int inputSize;
	double learningRate;
	double decayRate;
	vector<vector<double> > error;
	vector<vector<Neuron> > blocks;
	int getPreviousNeurons();
public:
	SerriformNetwork(int is, double l, double d);
	virtual ~SerriformNetwork();
	void addLayer(int size);
	vector<double> classify(vector<double> input);
	vector<double> train(vector<double> input, vector<double> target);
};

#endif /* SERRIFORMNETWORK_H_ */
