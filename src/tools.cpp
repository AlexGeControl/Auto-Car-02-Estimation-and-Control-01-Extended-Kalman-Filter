#include <iostream>
#include <cmath>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

/**
* Constructor.
*/
Tools::Tools() {}

/**
* Destructor.
*/
Tools::~Tools() {}

/**
* A helper method to calculate RMSE.
*/
VectorXd Tools::CalculateRMSE(
  const vector<VectorXd> &estimations,
  const vector<VectorXd> &ground_truth
) {
  // allocate RMSE:
  VectorXd rmse(4);
  // initialize:
	rmse << 0.0, 0.0, 0.0, 0.0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
    if (
        (estimations.size() > 0) && (estimations.size() == ground_truth.size())
    )
    {
    	//accumulate squared residuals
    	for(int i=0; i < estimations.size(); ++i){
            rmse += (estimations[i] - ground_truth[i]).array().square().matrix();
    	}

    	//calculate the mean
    	rmse /= estimations.size();

    	//calculate the squared root
    	rmse = rmse.array().sqrt();
    }

	//return the result
	return rmse;
}

/**
 * Calculate Jacobian for EKF correction
 * @param x The predicted state at k+1
 */
MatrixXd Tools::CalculateJacobian(const VectorXd& x) {
  // allocate Jacobian:
	MatrixXd Hj(3,4);

	// parse state parameters
	double px = x(0);
	double py = x(1);
	double vx = x(2);
	double vy = x(3);

  double r2 = pow(px, 2) + pow(py, 2);
	if (r2 > 0.0001) {
    //compute the Jacobian matrix
    Hj.block<1, 2>(0, 0) << px / pow(r2, 0.5), py / pow(r2, 0.5);
    Hj.block<1, 2>(0, 2) << 0.0, 0.0;
    Hj.block<1, 2>(1, 0) << -py / r2, px / r2;
    Hj.block<1, 2>(1, 2) << 0.0, 0.0;
    Hj.block<1, 2>(2, 0) << py * (vx * py - vy * px) / pow(r2, 1.5), px * (vy * px - vx * py) / pow(r2, 1.5);
    Hj.block<1, 2>(2, 2) << px / pow(r2, 0.5), py / pow(r2, 0.5);
	} else {
    cout << "[UpdateEKF]: Division by zero." << endl;
  }

	return Hj;
}

/**
 * Calculate predicted observation for EKF correction
 * @param x The predicted state at k+1
 */
VectorXd Tools::CalculatePredictedObservation(const VectorXd& x) {
  // allocate predicted observation:
	VectorXd y(3);

	// parse state parameters
	double px = x(0);
	double py = x(1);
	double vx = x(2);
	double vy = x(3);

  double r = sqrt(
    pow(px, 2) + pow(py, 2)
  );
  if (r > 0.01) {
    y << r,
         atan2(py, px),
         (px * vx + py * vy) / r;
  } else {
    cout << "[UpdateEKF]: Division by zero." << endl;
  }

	return y;
}
