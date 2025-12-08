# Preprocess 

This directory is responsible for processing all necessary from the IMU, cameras, and other data sources

## IMU Preintegration

## Outputs

- IMU Preintegration Factor:
    - delta_position
    - delta_velocity
    - covariance
    - timestamp_start
    - timestamp_end

- Each Feature Observation:
    - feature_id
    - pixel_coords
    - point_3d
    - is_static
    - covariance

## Authors
- Lucas Mitchell