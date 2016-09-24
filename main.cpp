/**
 *
 * A program to test a Serriform Neural Network
 * Author: Brandon Trabucco
 * Date: 2016/07/27
 *
 */

#include "SerriformNetwork.h"
#include "OutputTarget.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

long long getMSec() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

struct tm *getDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}

typedef struct {
	int inputSize = 6;
	int inputLength = 6;
	vector<vector<double> > sequence0 = {
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0} };
	int target0 = 0;

	vector<vector<double> > sequence1 = {
			{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 0.0, 1.0} };
	int target1 = 1;

	vector<vector<double> > sequence2 = {
			{0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
			{0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
			{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
			{1.0, 0.0, 0.0, 0.0, 0.0, 0.0} };
	int target2 = 2;

	vector<vector<double> > sequence3 = {
			{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
			{0.0, 0.0, 1.0, 0.0, 0.0, 0.0} };
	int target3 = 3;
} Dataset;

int main(int argc, char *argv[]) {
	cout << "Program initializing" << endl;
	if (argc < 3) {
		cout << argv[0] << " <learning rate> <decay rate> <size ...>" << endl;
		return -1;
	}

	int updatePoints = 100;
	int savePoints = 10;
	int maxEpoch = 1000;
	int sumNeurons = 0;
	double mse1 = 0, mse2 = 0, mse3 = 0, mse4 = 0;
	double learningRate = atof(argv[1]), decayRate = atof(argv[2]);
	long long networkStart, networkEnd, sumTime = 0;

	const int _day = getDate()->tm_mday;


	/**
	 *
	 * 	Open file streams to save data samples from Neural Network
	 * 	This data can be plotted via GNUPlot
	 *
	 */
	ostringstream errorDataFileName;
	errorDataFileName << "/u/trabucco/Desktop/Sequential_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-SNN-Error_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream errorData(errorDataFileName.str(), ios::app);
	if (!errorData.is_open()) return -1;

	ostringstream timingDataFileName;
	timingDataFileName << "/u/trabucco/Desktop/Sequential_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-SNN-Timing_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream timingData(timingDataFileName.str(), ios::app);
	if (!timingData.is_open()) return -1;

	ostringstream accuracyDataFileName;
	accuracyDataFileName << "/u/trabucco/Desktop/Sequential_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-SNN-Accuracy_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream accuracyData(accuracyDataFileName.str(), ios::app);
	if (!accuracyData.is_open()) return -1;

	ostringstream outputDataFileName;
	outputDataFileName << "/u/trabucco/Desktop/Sequential_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-SNN-Output_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream outputData(outputDataFileName.str(), ios::app);
	if (!outputData.is_open()) return -1;
	outputData << endl << endl;


	Dataset dataset;
	OutputTarget target(4, 4);
	cout << "Experiment loaded" << endl;


	SerriformNetwork network = SerriformNetwork(dataset.inputSize, learningRate, decayRate);
	cout << "Network initialized" << endl;


	for (int i = 0; i < (argc - 3); i++) {
		network.addLayer(atoi(argv[3 + i]));
		sumNeurons += atoi(argv[3 + i]);
	}  network.addLayer(4);


	int totalIterations = 0;
	bool converged = false;

	for (int e = 0; (e < maxEpoch); e++) {
		vector<double> error, output;

		networkStart = getMSec();
		for (int i = 0; i < dataset.inputLength; i++) {
			error = network.forward(dataset.sequence0[i], target.getOutputFromTarget(dataset.target0));
		} network.backward(); network.clear();

		mse1 = 0;
		for (int i = 0; i < error.size(); i++)
			mse1 += error[i] * error[i];
		mse1 /= error.size() * 2;

		for (int i = 0; i < dataset.inputLength; i++) {
			error = network.forward(dataset.sequence1[i], target.getOutputFromTarget(dataset.target1));
		} network.backward(); network.clear();

		mse2 = 0;
		for (int i = 0; i < error.size(); i++)
			mse2 += error[i] * error[i];
		mse2 /= error.size() * 2;

		for (int i = 0; i < dataset.inputLength; i++) {
			error = network.forward(dataset.sequence2[i], target.getOutputFromTarget(dataset.target2));
		} network.backward(); network.clear();

		mse3 = 0;
		for (int i = 0; i < error.size(); i++)
			mse3 += error[i] * error[i];
		mse3 /= error.size() * 2;

		for (int i = 0; i < dataset.inputLength; i++) {
			error = network.forward(dataset.sequence3[i], target.getOutputFromTarget(dataset.target3));
		} network.backward(); network.clear();

		mse4 = 0;
		for (int i = 0; i < error.size(); i++)
			mse4 += error[i] * error[i];
		mse4 /= error.size() * 2;

		int n = 0, c = 0;
		for (int i = 0; i < dataset.inputLength; i++) {
			output = network.forward(dataset.sequence0[i]);
			if (i == (dataset.inputLength - 1)) {
				n++;
				if (target.getTargetFromOutput(output) == dataset.target0) c++;
			}
		} network.clear(); for (int i = 0; i < dataset.inputLength; i++) {
			output = network.forward(dataset.sequence1[i]);
			if (i == (dataset.inputLength - 1)) {
				n++;
				if (target.getTargetFromOutput(output) == dataset.target1) c++;
			}
		} network.clear(); for (int i = 0; i < dataset.inputLength; i++) {
			output = network.forward(dataset.sequence2[i]);
			if (i == (dataset.inputLength - 1)) {
				n++;
				if (target.getTargetFromOutput(output) == dataset.target2) c++;
			}

		} network.clear(); for (int i = 0; i < dataset.inputLength; i++) {
			output = network.forward(dataset.sequence3[i]);
			if (i == (dataset.inputLength - 1)) {
				n++;
				if (target.getTargetFromOutput(output) == dataset.target3) c++;
			}
		} network.clear(); networkEnd = getMSec();

		sumTime += (networkEnd - networkStart);

		if (((e + 1) % (maxEpoch / updatePoints)) == 0) {
			cout << "Epoch " << e << " completed in " << (networkEnd - networkStart) << "msecs" << endl;
			cout << "Error[" << e << "] = " << ((mse1 + mse2 + mse3 + mse4) / 4) << endl;
			cout << "Accuracy[" << e << "] = " << (100.0 * (float)c / (float)n) << endl << endl;
		} errorData << e << ", " << ((mse1 + mse2 + mse3 + mse4) / 4) << endl;
		accuracyData << e << ", " << (100.0 * (float)c / (float)n) << endl;
	}

	timingData << sumNeurons << ", " << sumTime << ", " << totalIterations << endl;
	timingData.close();
	accuracyData.close();
	errorData.close();
	outputData.close();

	return 0;
}
