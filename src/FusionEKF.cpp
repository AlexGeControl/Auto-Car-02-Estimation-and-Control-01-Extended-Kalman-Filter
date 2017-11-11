#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225,      0,
                   0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09,      0,    0,
                 0, 0.0009,    0,
                 0,      0, 0.09;

  /**
    * Set the process and measurement noises
  */
  init_velocity_var = 1000.0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    cout << "[FusionEKF]: Initialization..." << endl;

    // first timestamp:
    previous_timestamp_ = measurement_pack.timestamp_;

    // first measurement:
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
        * Convert radar from polar to cartesian coordinates and initialize state.
        */
        // parse radar measurements:
        double ro, theta, ro_dot;

        ro = measurement_pack.raw_measurements_(0);
        theta = measurement_pack.raw_measurements_(1);
        ro_dot = measurement_pack.raw_measurements_(2);

        // position:
        ekf_.x_(0) =  ro * cos(theta);
        ekf_.x_(1) =  ro * sin(theta);
        // velocity:
        ekf_.x_(2) =  ro_dot * cos(theta);
        ekf_.x_(3) =  ro_dot * sin(theta);

        // set initial state covariance matrix according to radar measurement noise:
        ekf_.P_ << R_radar_(0, 0),         0.0000,         0.0000,         0.0000,
                           0.0000, R_radar_(0, 0),         0.0000,         0.0000,
                           0.0000,         0.0000, R_radar_(2, 2),         0.0000,
                           0.0000,         0.0000,         0.0000, R_radar_(2, 2);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
        * Initialize state from lidar measurements:
        */
        double px, py;

        px = measurement_pack.raw_measurements_(0);
        py = measurement_pack.raw_measurements_(1);

        // position:
        ekf_.x_(0) = px;
        ekf_.x_(1) = py;
        // velocity:
        ekf_.x_(2) = 0.0;
        ekf_.x_(3) = 0.0;

        // set initial state covariance matrix according to lidar measurement noise:
        ekf_.P_ << R_laser_(0, 0),         0.0000,            0.0000,            0.0000,
                           0.0000, R_laser_(0, 0),            0.0000,            0.0000,
                           0.0000,         0.0000, init_velocity_var,            0.0000,
                           0.0000,         0.0000,            0.0000, init_velocity_var;
    }

    cout << "[FusionEKF]: Initialization done." << endl;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  // parse current timestamp:
  long long current_timestamp = measurement_pack.timestamp_;
  long long delta_T = current_timestamp - previous_timestamp_;
  previous_timestamp_ = current_timestamp;

  ekf_.Predict(delta_T / 1000000.0);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */
  // parse measurementsL
  const VectorXd &z = measurement_pack.raw_measurements_;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Set up measurement params:
    ekf_.R_ = R_radar_;
    // Radar updates
    ekf_.UpdateEKF(z);
  } else {
    // Set up measurement params:
    ekf_.R_ = R_laser_;
    // Laser updates
    ekf_.Update(z);
  }
}
