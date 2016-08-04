/*
 * SawtoothNetwork.h
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#ifndef SERRIFORMNETWORK_H_
#define SERRIFORMNETWORK_H_

#include <vector>
#include <omp.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include "Neuron.h"
using namespace std;

class SerriformNetwork {
private:
	int inputSize;
	int overlap;
	double learningRate;
	double decayRate;
	double **error;
	vector<vector<Neuron> > layers;
	int getPreviousNeurons();
	int getPreviousNeurons(int l);
public:
	SerriformNetwork(int is, int o, double l, double d);
	virtual ~SerriformNetwork();
	void addLayer(int size);
	vector<double> classify(vector<double> input);
	vector<double> train(vector<double> input, vector<double> target);
};

#endif /* SERRIFORMNETWORK_H_ */
