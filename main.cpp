/**
 *
 * A program to test a Serriform Neural Network
 * Author: Brandon Trabucco
 * Date: 2016/07/27
 *
 */

#include "SerriformNetwork.h"
#include "DatasetAdapter.h"
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


int main(int argc, char *argv[]) {
	cout << "Program initializing" << endl;
	if (argc < 5) {
		cout << argv[0] << " <learning rate> <decay rate> <serr. width> <serr. depth>" << endl;
		return -1;
	}

	int updatePoints = 100;
	int savePoints = 10;
	int maxEpoch = 100;
	int sumNeurons = 0;
	double mse;
	double learningRate = atof(argv[1]), decayRate = atof(argv[2]);
	int width = atoi(argv[3]), depth = atoi(argv[4]);
	long long networkStart, networkEnd, sumTime = 0;

	const int _day = getDate()->tm_mday;


	/**
	 *
	 * 	Open file streams to save data samples from Neural Network
	 * 	This data can be plotted via GNUPlot
	 *
	 */
	ostringstream learningDataFileName;
	learningDataFileName << "/stash/tlab/trabucco/ANN_Saves/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Multicore-SNN-Data_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";

	ostringstream errorDataFileName;
	errorDataFileName << "/u/trabucco/Desktop/KTH_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-SNN-Error_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream errorData(errorDataFileName.str(), ios::app);
	if (!errorData.is_open()) return -1;

	ostringstream timingDataFileName;
	timingDataFileName << "/u/trabucco/Desktop/KTH_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-SNN-Timing_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream timingData(timingDataFileName.str(), ios::app);
	if (!timingData.is_open()) return -1;

	ostringstream accuracyDataFileName;
	accuracyDataFileName << "/u/trabucco/Desktop/KTH_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-SNN-Accuracy_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream accuracyData(accuracyDataFileName.str(), ios::app);
	if (!accuracyData.is_open()) return -1;

	ostringstream outputDataFileName;
	outputDataFileName << "/u/trabucco/Desktop/KTH_Convergence_Data_Files/" <<
			(getDate()->tm_year + 1900) << "-" << (getDate()->tm_mon + 1) << "-" << _day <<
			"_Single-Core-SNN-Output_" << learningRate <<
			"-learning_" << decayRate << "-decay.csv";
	ofstream outputData(outputDataFileName.str(), ios::app);
	if (!outputData.is_open()) return -1;
	outputData << endl << endl;


	DatasetAdapter dataset = DatasetAdapter();
	OutputTarget target(6);
	cout << "Experiment loaded" << endl;


	SerriformNetwork network = SerriformNetwork(dataset.getFrameSize(), learningRate, decayRate);
	cout << "Network initialized" << endl;


	for (int i = 0; i < depth; i++) {
		network.addLayer(width);
		sumNeurons += width;
	}  network.addLayer(6);


	int totalIterations = 0;
	bool converged = false;

	for (int e = 0; (e < maxEpoch); e++) {
		vector<double> error, output;

		networkStart = getMSec();
		while (dataset.nextTrainingVideo()) {
			while (dataset.nextTrainingFrame()) {
				DatasetExample ex = dataset.getTrainingFrame();
				error = network.forward(ex.frame, target.getOutputFromTarget(ex.label));
			} network.backward(); network.clear();
		} dataset.reset();

		mse = 0;
		for (int i = 0; i < error.size(); i++)
			mse += error[i] * error[i];
		mse /= error.size() * 2;

		long long n = 0, c = 0;
		while (dataset.nextTestVideo()) {
			while (dataset.nextTestFrame()) {
				DatasetExample ex = dataset.getTestFrame();
				output = network.forward(ex.frame);
				n++;
				if (target.getTargetFromOutput(output) == ex.label) c++;
			} network.clear();
		} dataset.reset();

		networkEnd = getMSec();
		sumTime += (networkEnd - networkStart);

		if (((e + 1) % (maxEpoch / updatePoints)) == 0) {
			cout << "Epoch " << e << " completed in " << (networkEnd - networkStart) << "msecs" << endl;
			cout << "Error[" << e << "] = " << (mse) << endl;
			cout << "Accuracy[" << e << "] = " << (100.0 * (float)c / (float)n) << endl << endl;
			if (((e + 1) % (maxEpoch / savePoints)) == 0) network.saveToFile(learningDataFileName.str());
		} errorData << e << ", " << (mse) << endl;
		accuracyData << e << ", " << (100.0 * (float)c / (float)n) << endl;
	}

	timingData << sumNeurons << ", " << sumTime << ", " << totalIterations << endl;
	timingData.close();
	accuracyData.close();
	errorData.close();
	outputData.close();

	return 0;
}
