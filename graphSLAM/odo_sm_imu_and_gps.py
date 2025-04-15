"""
Integrate IMU and GPS
"""
import gtsam
from gtsam import symbol
import numpy as np

# === Setup ===
g = 9.81
imu_params = gtsam.PreintegrationParams.MakeSharedU(g)
imu_params.setAccelerometerCovariance(np.eye(3)*0.1)
imu_params.setGyroscopeCovariance(np.eye(3)*0.1)
imu_params.setIntegrationCovariance(np.eye(3)*0.01)

# Noise models
POSE_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))       # For odometry & LiDAR
IMU_BIAS_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
GPS_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
VEL_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

# ISAM2 setup
isam = gtsam.ISAM2()
bias = gtsam.imuBias.ConstantBias()

# Initial state
key_pose = lambda i: symbol('x', i)
key_vel = lambda i: symbol('v', i)
key_bias = lambda i: symbol('b', i)

pose0 = gtsam.Pose3()
vel0 = np.zeros(3)
state0 = gtsam.NavState(pose0, vel0)

initial = gtsam.Values()
initial.insert(key_pose(0), pose0)
initial.insert(key_vel(0), vel0)
initial.insert(key_bias(0), bias)

graph = gtsam.NonlinearFactorGraph()
graph.add(gtsam.PriorFactorPose3(key_pose(0), pose0, POSE_NOISE))
graph.add(gtsam.PriorFactorVector(key_vel(0), vel0, VEL_NOISE))
graph.add(gtsam.PriorFactorConstantBias(key_bias(0), bias, IMU_BIAS_NOISE))

isam.update(graph, initial)

# === Main loop ===
state = state0
imu_preint = gtsam.PreintegratedImuMeasurements(imu_params, bias)

for i in range(1, 20):
    # 1. Integrate IMU
    dt = 0.01
    for _ in range(100):  # 1 second @ 100 Hz
        accel = np.array([0, 0, g]) + np.random.randn(3) * 0.02
        gyro = np.random.randn(3) * 0.005
        imu_preint.integrateMeasurement(accel, gyro, dt)

    # Predict new pose/vel
    predicted_state = imu_preint.predict(state, bias)

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # 2. IMU factor
    graph.add(gtsam.ImuFactor(
        key_pose(i-1), key_vel(i-1),
        key_pose(i), key_vel(i),
        key_bias(i-1), imu_preint))

    # 3. Bias evolution
    graph.add(gtsam.BetweenFactorConstantBias(
        key_bias(i-1), key_bias(i),
        gtsam.imuBias.ConstantBias(), IMU_BIAS_NOISE))

    # 4. GPS factor (simulated)
    gps_meas = predicted_state.pose().translation() + np.random.randn(3) * 0.5
    graph.add(gtsam.GPSFactor(key_pose(i), gps_meas, GPS_NOISE))

    # 5. Wheel Odometry (BetweenFactorPose3)
    delta_wheel = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.01, 0.005, 0), np.array([1.0, 0.0, 0.0]))
    graph.add(gtsam.BetweenFactorPose3(
        key_pose(i-1), key_pose(i), delta_wheel, POSE_NOISE))

    # 6. LiDAR Scan Matching Pose (BetweenFactorPose3)
    delta_lidar = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.015, 0.003, 0.001), np.array([1.02, 0.02, -0.01]))
    graph.add(gtsam.BetweenFactorPose3(
        key_pose(i-1), key_pose(i), delta_lidar, POSE_NOISE))

    # 7. Initial guesses
    values.insert(key_pose(i), predicted_state.pose())
    values.insert(key_vel(i), predicted_state.velocity())
    values.insert(key_bias(i), bias)

    # Update ISAM2
    isam.update(graph, values)
    isam.update()

    # Get result
    result = isam.calculateEstimate()
    pose_est = result.atPose3(key_pose(i))
    vel_est = result.atVector(key_vel(i))
    bias = result.atConstantBias(key_bias(i))

    print(f"Step {i}: pose = {pose_est.translation()}")

    # Prepare for next loop
    imu_preint.resetIntegrationAndSetBias(bias)
    state = gtsam.NavState(pose_est, vel_est)