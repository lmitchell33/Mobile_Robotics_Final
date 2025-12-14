import numpy as np
import gtsam

# TODO: Update the covar values
def imu_preintegration_factory(bias: gtsam.imuBias.ConstantBias, gravity=9.81, gryo_cov=1e-3, accel_cov=1e-2, integration_cov=1e-4):
    """ Factor method to create an instance of my IMU Preintegrator class"""
    params = gtsam.PreintegrationParams.MakeSharedU(gravity)
    params.n_gravity = np.array([0.0, 0.0, -gravity])
    params.setGyroscopeCovariance(np.eye(3) * gryo_cov)
    params.setAccelerometerCovariance(np.eye(3) * accel_cov)
    params.setIntegrationCovariance(np.eye(3) * integration_cov)
    return IMUPreintegrator(params, bias)

class IMUPreintegrator:
    """Pretty much just a wrapper for the gtsam.PreintegratedImuMeasurements class"""
    def __init__(self, preint_params: gtsam.PreintegrationParams, bias: gtsam.imuBias.ConstantBias):
        self._params = preint_params
        self._bias = bias
        self.preintegration = gtsam.PreintegratedImuMeasurements(self._params, self._bias)

    def __str__(self):
        delta_p, delta_v, delta_r = self.get_deltas()
        return f"IMU Preintegrator: \n Change in Position: {delta_p}\n Change in Velocity: {delta_v}\n Change in Rotation: {delta_r}"

    def integrate_measurement(self, accel: np.ndarray, omega: np.ndarray, dt: float):
        self.preintegration.integrateMeasurement(accel, omega, dt)

    def reset(self):
        self.preintegration = gtsam.PreintegratedImuMeasurements(self._params, self._bias)

    def set_bias(self, new_bias):
        self._bias = new_bias

    def get_deltas(self):
        delta_position = self.preintegration.deltaPij()
        delta_velocity = self.preintegration.deltaVij()
        delta_rotation = self.preintegration.deltaRij()
        return delta_position, delta_velocity, delta_rotation


if __name__ == "__main__":
    # just some code to quickly test it runs correctly
    preintegrator = imu_preintegration_factory(gtsam.imuBias.ConstantBias())

    # test at 100 Hz (0.01 second between measurements)
    dt = 0.01
    for i in range(10):
        # move forward slowly 
        test_accel = np.array([1.0, 1.7, -9.81])
        test_omega = np.array([1.3, 0.67, 0.0])
        preintegrator.integrate_measurement(test_accel, test_omega, dt)
    
    print(preintegrator)