#include <iostream>
#include <cmath>
#include "tools.h"
#include "kalman_filter.h"

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Please note that the Eigen library does not initialize
 * VectorXd or MatrixXd objects with zeros upon creation.
 */

/**
* Constructor
*/
KalmanFilter::KalmanFilter() {
  // allocate state vector:
  x_ = VectorXd(4);
  // allocate state covariance matrix:
  P_ = MatrixXd(4, 4);

  // allocate system matrix:
  F_ = MatrixXd(4, 4);
  F_ << 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0;

  // allocate system noise covariance matrix:
  Q_ = MatrixXd(4, 4);

  // identity matrix:
  I_ = MatrixXd::Identity(4, 4);
}

/**
 * Destructor
 */
KalmanFilter::~KalmanFilter() {}

/**
 * Init Initializes Kalman filter
 *
 * @param x_in Initial state
 * @param P_in Initial state covariance
 *
 * @param F_in Transition matrix
 * @param Q_in Process covariance matrix
 *
 * @param H_in Measurement matrix
 * @param R_in Measurement covariance matrix
 */
void KalmanFilter::Init(
  VectorXd &x_in, MatrixXd &P_in,
  MatrixXd &F_in, MatrixXd &Q_in,
  MatrixXd &H_in, MatrixXd &R_in
) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;
  H_ = H_in;
  R_ = R_in;
}

/**
 * Prediction Predicts the state and the state covariance
 * using the process model
 * @param delta_T Time between k and k+1 in s
 */
void KalmanFilter::Predict(
  const double delta_T,
  const double noise_ax,
  const double noise_ay
) {
  // set up system matrix F:
  F_(0, 2) = F_(1, 3) = delta_T;

  // set up system noise covariance matrix:
  for (int i = 0; i < 2; ++i) {
	    for (int j = 0; j < 2; ++j) {
	        Q_.block<2, 2>(i << 1, j << 1) << noise_ax, 0.0, 0.0, noise_ay;
	    }
	}
	Q_.block<2, 2>(0, 0) *= 1.0/4.0 * pow(delta_T, 4);
	Q_.block<2, 2>(0, 2) *= 1.0/2.0 * pow(delta_T, 3);
	Q_.block<2, 2>(2, 0) *= 1.0/2.0 * pow(delta_T, 3);
	Q_.block<2, 2>(2, 2) *= pow(delta_T, 2);

  // predict state vector:
  x_ = F_ * x_;
  // predict state covariance matrix:
  P_ = F_ * P_ * F_.transpose() + Q_;
}

/**
 * Updates the state by using standard Kalman Filter equations
 * @param z The measurement at k+1
 */
void KalmanFilter::Update(const VectorXd &z) {
  // measurement matrix - laser
  H_ = MatrixXd(2, 4);
  H_ << 1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0;

  // Kalman gain:
  MatrixXd Ht_ = H_.transpose();
  MatrixXd K_ = P_ * Ht_ * (H_ * P_ * Ht_ + R_).inverse();

  // update state vector:
  x_ += K_ * (z - H_ * x_);
  // update state covariance matrix:
  P_ = (I_ - K_ * H_) * P_;
}

/**
 * Updates the state by using Extended Kalman Filter equations
 * @param z The measurement at k+1
 */
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // linearized measurement matrix - radar:
  H_ = tools.CalculateJacobian(x_);
  // predicted observation:
  VectorXd y = tools.CalculatePredictedObservation(x_);

  // Kalman gain:
  MatrixXd Ht_ = H_.transpose();
  MatrixXd K_ = P_ * Ht_ * (H_ * P_ * Ht_ + R_).inverse();

  // update state vector:
  VectorXd error = z - y;
  if (error(1) > M_PI) {
    error(1) = 2*M_PI - error(1);
  } else if (error(1) < -M_PI){
    error(1) = 2*M_PI + error(1);
  }
  x_ += K_ * error;
  // update state covariance matrix:
  P_ = (I_ - K_ * H_) * P_;
}
