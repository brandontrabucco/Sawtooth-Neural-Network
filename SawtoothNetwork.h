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
	vector<vector<Neuron> > layers;
	int getPreviousNeurons();
public:
	SawtoothNetwork(int is);
	virtual ~SawtoothNetwork();
	void addLayer(int size);
	vector<double> feedforward(vector<double> input);
};

#endif /* SAWTOOTHNETWORK_H_ */
