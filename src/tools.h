#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(
    const vector<VectorXd> &estimations,
    const vector<VectorXd> &ground_truth
  );

  /**
   * Calculate Jacobian for EKF correction
   * @param x The predicted state at k+1
   */
  MatrixXd CalculateJacobian(const VectorXd& x);

  /**
   * Calculate predicted observation for EKF correction
   * @param x The predicted state at k+1
   */
  VectorXd CalculatePredictedObservation(const VectorXd& x);
};

#endif /* TOOLS_H_ */
