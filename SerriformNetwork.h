/*
 * SawtoothNetwork.h
 *
 *  Created on: Jul 27, 2016
 *      Author: trabucco
 */

#ifndef SERRIFORMNETWORK_H_
#define SERRIFORMNETWORK_H_

#include <vector>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <omp.h>
#include "Neuron.h"
using namespace std;

class SerriformNetwork {
private:
	int timestep;
	int inputSize;
	double learningRate;
	double decayRate;
	vector<vector<double> > error;
	vector<vector<vector<double> > > errorBuffer;
	vector<vector<Neuron> > layers;
	int getPreviousNeurons();
	int getPreviousNeurons(int l);
public:
	SerriformNetwork(int is, double l, double d);
	virtual ~SerriformNetwork();
	void addLayer(int size);
	vector<double> forward(vector<double> input);
	vector<double> forward(vector<double> input, vector<double> target);
	void backward();
	void clear();
	vector<double> train(vector<double> input, vector<double> target);
};

#endif /* SERRIFORMNETWORK_H_ */
