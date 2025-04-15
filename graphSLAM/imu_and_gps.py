"""
Integrate IMU and GPS
"""
import gtsam
from gtsam import symbol
import numpy as np

# === IMU parameters ===
g = 9.81
imu_params = gtsam.PreintegrationParams.MakeSharedU(g)
imu_params.setAccelerometerCovariance(np.eye(3) * 0.1)
imu_params.setGyroscopeCovariance(np.eye(3) * 0.1)
imu_params.setIntegrationCovariance(np.eye(3) * 0.01)

# Noise models
POSE_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))  # x,y,z,roll,pitch,yaw
VEL_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
BIAS_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
GPS_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)  # 1 meter accuracy

# ISAM2
isam = gtsam.ISAM2()

# Initial values
new_factors = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

# Initial state
pose0 = gtsam.Pose3()
vel0 = np.zeros(3)
bias0 = gtsam.imuBias.ConstantBias()

state0 = gtsam.NavState(pose0, vel0)
key_pose = lambda i: symbol('x', i)
key_vel = lambda i: symbol('v', i)
key_bias = lambda i: symbol('b', i)

# Add priors
initial.insert(key_pose(0), pose0)
initial.insert(key_vel(0), vel0)
initial.insert(key_bias(0), bias0)

new_factors.add(gtsam.PriorFactorPose3(key_pose(0), pose0, POSE_NOISE))
new_factors.add(gtsam.PriorFactorVector(key_vel(0), vel0, VEL_NOISE))
new_factors.add(gtsam.PriorFactorConstantBias(key_bias(0), bias0, BIAS_NOISE))

# ISAM2 update
isam.update(new_factors, initial)

# === IMU + GPS loop ===
current_state = state0
current_bias = bias0
imu_preint = gtsam.PreintegratedImuMeasurements(imu_params, current_bias)

for i in range(1, 20):
    # Simulate IMU data (could be from sensor or bag file)
    dt = 0.01  # IMU rate
    for _ in range(100):  # 100 Hz x 1 sec
        accel = np.array([0, 0, g]) + np.random.randn(3) * 0.01
        gyro = np.array([0, 0, 0]) + np.random.randn(3) * 0.001
        imu_preint.integrateMeasurement(accel, gyro, dt)

    # Predict pose/velocity
    new_state = imu_preint.predict(current_state, current_bias)

    # Create graph for this timestep
    new_factors = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Add IMU factor
    new_factors.add(gtsam.ImuFactor(
        key_pose(i - 1), key_vel(i - 1),
        key_pose(i), key_vel(i),
        key_bias(i - 1), imu_preint
    ))

    # Bias evolution
    new_factors.add(gtsam.BetweenFactorConstantBias(
        key_bias(i - 1), key_bias(i),
        gtsam.imuBias.ConstantBias(), BIAS_NOISE
    ))

    # GPS measurement (simulated)
    gps_position = new_state.pose().translation() + np.random.randn(3) * 0.5
    new_factors.add(gtsam.GPSFactor(key_pose(i), gps_position, GPS_NOISE))

    # Initial guesses
    initial.insert(key_pose(i), new_state.pose())
    initial.insert(key_vel(i), new_state.velocity())
    initial.insert(key_bias(i), current_bias)

    # ISAM2 update
    isam.update(new_factors, initial)

    # Get result
    result = isam.calculateEstimate()
    current_state = gtsam.NavState(
        result.atPose3(key_pose(i)),
        result.atVector(key_vel(i))
    )
    current_bias = result.atConstantBias(key_bias(i))
    imu_preint.resetIntegrationAndSetBias(current_bias)

    print(f"Pose {i}: {result.atPose3(key_pose(i)).translation()}")
