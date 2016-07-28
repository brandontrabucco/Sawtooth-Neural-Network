/*
 * SawtoothNetwork.h
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#ifndef SAWTOOTHNETWORK_H_
#define SAWTOOTHNETWORK_H_

#include <vector>
#include "Neuron.h"
using namespace std;

class SawtoothNetwork {
private:
	int inputSize;
	double learningRate;
	double decayRate;
	vector<vector<double> > error;
	vector<vector<Neuron> > layers;
	int getPreviousNeurons();
public:
	SawtoothNetwork(int is, double l, double d);
	virtual ~SawtoothNetwork();
	void addLayer(int size);
	vector<double> classify(vector<double> input);
	vector<double> train(vector<double> input, vector<double> target);
};

#endif /* SAWTOOTHNETWORK_H_ */
