/**
 *
 * A program to test a Sawtooth Neural Network
 * Author: Brandon Trabucco
 * Date: 2016/07/27
 *
 */

#include "SawtoothNetwork.h"
#include <vector>
#include <iostream>
using namespace std;

typedef struct {
	int inputSize = 6;
	int inputLength = 6;
	vector<vector<double> > list = {
			{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
	vector<double> operator[](unsigned int i) {
		if (i < list.size()) return list[i];
		else return list[0];
	}
} Dataset;

int main() {
	Dataset dataset;
	SawtoothNetwork network = SawtoothNetwork(dataset.inputSize, 1.0, 1.0);

	network.addLayer(6);
	network.addLayer(3);
	network.addLayer(1);

	for (int i = 0; i < dataset.inputLength; i++) {
		cout << "t = " << i << endl;
		vector<double> output = network.classify(dataset[i]);
		for (int j = 0; j < output.size(); j++) {
			cout << "output[" << j << "] = " << output[j] << endl;
		} cout << endl;
	}

	_fgetchar();
	return 0;
}
