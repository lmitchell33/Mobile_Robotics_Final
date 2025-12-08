import os, sys
os.add_dll_directory(r"C:\Users\Nate\Documents\gtsam\build\bin\Release")
sys.path.append(r"C:\Users\Nate\Documents\gtsam\build\python")


import numpy as np
import gtsam




def X(i): return gtsam.symbol('x', i)  # pose
def V(i): return gtsam.symbol('v', i)  # velocity
def B(i): return gtsam.symbol('b', i)  # bias
def L(fid): return gtsam.symbol('l', fid)  # landmark

def rotmat_to_quat(R):
    """Return quaternion [w, x, y, z] from a 3×3 rotation matrix."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S

    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S

    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S

    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    return np.array([w, x, y, z])

class BuildFactor:

    def __init__(self, window_size, fps, fx, fy, s, cx, cy):

        # Use BatchFixedLagSmoother — correct for your GTSAM build
        self.smoother = gtsam.BatchFixedLagSmoother(window_size)
        self.smoother_estimate = gtsam.Values()

        # camera calibration
        self.cam = gtsam.Cal3_S2(fx, fy, s, cx, cy)

        # internal index
        self.frame_idx = -1

        # bookkeeping
        self.landmark_info = {}
        self.virtual_landmarks = {}

        # noise models
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]))
        self.prior_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.prior_bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        self.reproj_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self.huber_k = 1.345

        # dt
        self.dt = 1/fps
        self.initialized = False

    def initialize_first_pose(self, pose0: gtsam.Pose3,vel0=None):

        if self.initialized:
            raise RuntimeError("Already initialized")

        self.frame_idx = 0

        vel0 = np.zeros(3) if vel0 is None else vel0
        bias0 = gtsam.imuBias.ConstantBias()

        # Create initial factors + values
        factors = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        timestamps = {}

        # time stamp for pose 0
        timestamps[X(0)] = 0
        timestamps[V(0)] = 0
        timestamps[B(0)] = 0

        # priors
        factors.add(gtsam.PriorFactorPose3(X(0), pose0, self.prior_pose_noise))
        factors.add(gtsam.PriorFactorPoint3(V(0), gtsam.Point3(*vel0), self.prior_vel_noise))
        factors.add(gtsam.PriorFactorConstantBias(B(0), bias0, self.prior_bias_noise))

        # initial values
        initial.insert(X(0), pose0)
        initial.insert(V(0), gtsam.Point3(*vel0))
        initial.insert(B(0), bias0)

        # update smoother
        self.smoother.update(factors, initial, timestamps)
        self.smoother_estimate = self.smoother.calculateEstimate()

        self.initialized = True

    def add_imu_factor(self, preint_imu: gtsam.PreintegratedImuMeasurements,timestamp: float):

        if not self.initialized:
            raise RuntimeError("Call initialize_first_pose() first")

        i = self.frame_idx
        j = i + 1
        self.frame_idx = j

        factors = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        timestamps = {}

        pose_key_i, vel_key_i, bias_key_i = X(i), V(i), B(i)
        pose_key_j, vel_key_j, bias_key_j = X(j), V(j), B(j)


        timestamps[pose_key_j] = timestamp
        timestamps[vel_key_j] = timestamp
        timestamps[bias_key_j] = timestamp

        prev_pose = self.smoother_estimate.atPose3(pose_key_i)
        prev_vel = self.smoother_estimate.atPoint3(vel_key_i)
        prev_bias = self.smoother_estimate.atConstantBias(bias_key_i)

        initial.insert(pose_key_j, prev_pose)
        initial.insert(vel_key_j, prev_vel)
        initial.insert(bias_key_j, prev_bias)


        imu_factor = gtsam.ImuFactor(
            pose_key_i, vel_key_i,
            pose_key_j, vel_key_j,
            bias_key_i,
            preint_imu
        )
        factors.add(imu_factor)

        # bias random walk
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        factors.add(gtsam.BetweenFactorConstantBias(
            bias_key_i, bias_key_j,
            gtsam.imuBias.ConstantBias(),
            bias_noise
        ))

        # update the fixed-lag smoother
        self.smoother.update(factors, initial, timestamps)
        self.smoother_estimate = self.smoother.calculateEstimate()

    def _add_projection_factor(self, factors, pose_key, landmark_key, uv, noise_sigma):
        meas = gtsam.Point2(float(uv[0]), float(uv[1]))
        base_noise = gtsam.noiseModel.Isotropic.Sigma(2, noise_sigma)
        robust = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(self.huber_k),
            base_noise
        )
        proj = gtsam.GenericProjectionFactorCal3_S2(
            meas, robust, pose_key, landmark_key, self.cam
        )
        factors.add(proj)

    def add_feature_observations(self, observations, timestamp: float):
        """
        observations must contain:
            obs.feature_id
            obs.pixel_coords
            obs.point_3d  (optional cam frame 3D)
            obs.is_static
            obs.uncertainty (optional)
        """
        j = self.frame_idx
        pose_key = X(j)

        factors = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        timestamps = {}

        for obs in observations:
            fid = int(obs.feature_id)
            lkey = L(fid)

            # register feature info
            if fid not in self.landmark_info:
                self.landmark_info[fid] = {"is_static": obs.is_static}

            # initialize landmark if not in smoother estimate
            if not self.smoother_estimate.exists(lkey):
                if obs.point_3d is None:
                    continue  # need triangulation before adding
                # convert to world frame
                pose_j = self.smoother_estimate.atPose3(pose_key)
                Xc, Yc, Zc = obs.point_3d
                pw = pose_j.transformFrom(gtsam.Point3(Xc, Yc, Zc))
                initial.insert(lkey, pw)
                timestamps[lkey] = timestamp

                # optional small prior for static landmarks
                if obs.is_static:
                    if getattr(obs, "uncertainty", None) is not None:
                        cov = np.array(obs.uncertainty)
                        sigmas = np.sqrt(np.diag(cov))
                        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
                    else:
                        prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
                    factors.add(gtsam.PriorFactorPoint3(lkey, pw, prior_noise))

            # always add projection factor
            self._add_projection_factor(factors, pose_key, lkey,
                                        obs.pixel_coords,
                                        noise_sigma=1.0 if obs.is_static else 2.0)

        # update smoother
        self.smoother.update(factors, initial, timestamps)
        self.smoother_estimate = self.smoother.calculateEstimate()

    def add_virtual_landmark(self, feature_id, predicted_world, uv, timestamp):
        fid = int(feature_id)
        lkey = L(fid)

        factors = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        timestamps = {}

        self.virtual_landmarks[fid] = predicted_world.copy()

        # initialize virtual landmark if needed
        if not self.smoother_estimate.exists(lkey):
            pt = gtsam.Point3(*predicted_world)
            initial.insert(lkey, pt)
            timestamps[lkey] = timestamp

        # Weak prior to avoid underconstrained system
            weak_prior = gtsam.noiseModel.Isotropic.Sigma(3, 5.0)
            factors.add(gtsam.PriorFactorPoint3(lkey, pt, weak_prior))

        # Add projection factor
        pose_key = X(self.frame_idx)
        self._add_projection_factor(factors, pose_key, lkey, uv, noise_sigma=2.0)

        # update smoother
        self.smoother.update(factors, initial, timestamps)
        self.smoother_estimate = self.smoother.calculateEstimate()

#---------------------------Output Functions ------------------------------
    def get_pose(self, idx):
        key = X(idx)
        if self.smoother_estimate.exists(key):
            return self.smoother_estimate.atPose3(key)
        return None

    def get_current_trajectory(self):
        traj = []
        for key in self.smoother_estimate.keys():
            if gtsam.Symbol(key).chr() == ord('x'):
                idx = gtsam.Symbol(key).index()
                pose = self.smoother_estimate.atPose3(key)

                # Translation (numpy array)
                t = pose.translation()        # returns array [x, y, z]
                x, y, z = t[0], t[1], t[2]

                # Rotation matrix (always works)
                R = pose.rotation().matrix()

                # Convert to quaternion (your helper)
                q = rotmat_to_quat(R)  # returns [w, x, y, z]

                traj.append(
                    (idx, np.array([x, y, z, *q]))
                )

        traj.sort()
        return traj

    def get_landmarks(self):
        out = {}
        for key in self.smoother_estimate.keys():
            if gtsam.Symbol(key).chr() == ord('l'):
                fid = gtsam.Symbol(key).index()

                p = self.smoother_estimate.atPoint3(key)  # returns numpy array [x, y, z]

                x, y, z = p[0], p[1], p[2]

                out[fid] = np.array([x, y, z])

        return out

    