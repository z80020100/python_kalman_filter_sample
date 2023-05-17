#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilterOneDimension


SAMPLE_DRIVING_DATA_FILE_NAME = "driving_data.csv"
OUTPUT_FILE_NAME = "driving_data_filtered.csv"


def main():
    with open(SAMPLE_DRIVING_DATA_FILE_NAME) as f:
        reader = csv.reader(f)
        data = list(reader)

    # Get the index of each column
    columns_name = data[0]
    for i in range(len(columns_name)):
        column_name = columns_name[i]
        if "Time" in column_name:
            time_idx = i
        elif "EngineSpeed" in column_name:
            engine_speed_idx = i
        elif "AcceleratorOpeningAngle" in column_name:
            accelerator_opening_angle_idx = i
        elif "TurnSignal" in column_name:
            turn_signal_idx = i
        elif "SteeringAngle" in column_name:
            steering_angle_idx = i
        elif "Speed" in column_name:
            speed_idx = i
        elif "BrakeOilPressure" in column_name:
            brake_oil_pressure_idx = i
        elif "YawRate" in column_name:
            yaw_rate_idx = i
        elif "ForwardAndRearwardG" in column_name:
            forward_and_rearward_g_idx = i
        elif "LateralG" in column_name:
            lateral_g_idx = i
        else:
            pass

    time = []
    engine_speed_measure = []
    accelerator_opening_angle_measure = []
    turn_signal_measure = []
    steering_angle_measure = []
    speed_measure = []
    brake_oil_pressure_measure = []
    yaw_rate_measure = []
    forward_and_rearward_g_measure = []
    lateral_g_measure = []

    for i in range(1, len(data)):
        time.append(float(data[i][time_idx]))
        engine_speed_measure.append(float(data[i][engine_speed_idx]))
        accelerator_opening_angle_measure.append(float(data[i][accelerator_opening_angle_idx]))
        turn_signal_measure.append(float(data[i][turn_signal_idx]))
        steering_angle_measure.append(float(data[i][steering_angle_idx]))
        speed_measure.append(float(data[i][speed_idx]))
        brake_oil_pressure_measure.append(float(data[i][brake_oil_pressure_idx]))
        yaw_rate_measure.append(float(data[i][yaw_rate_idx]))
        forward_and_rearward_g_measure.append(float(data[i][forward_and_rearward_g_idx]))
        lateral_g_measure.append(float(data[i][lateral_g_idx]))

    speed_estimate = []
    # Set Kalman filter parameters
    state_transition_matrix = 1.0 # consider the current state as the next state (no change)
    observation_matrix = 1.0 # consider no scaling between the state and the measurement
    process_noise_matrix = 0.05 # guessing value
    measurement_noise = 0.5 # guessing value
    estimated_error = 0.0 # according to the initialization estimate error
    control_matrix = 0.0 # no control input
    measurable_input = 0.0 # no measurable input
    kf = KalmanFilterOneDimension(speed_measure[0], state_transition_matrix,
                                  observation_matrix, process_noise_matrix,
                                  measurement_noise, estimated_error,
                                  control_matrix, measurable_input)
    for i in range(len(speed_measure)):
        speed_estimate.append(kf.execute(speed_measure[i]))

    with open(OUTPUT_FILE_NAME, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["Time", "EngineSpeed", "AcceleratorOpeningAngle",
                         "TurnSignal", "SteeringAngle", "Speed",
                         "BrakeOilPressure", "YawRate", "ForwardAndRearwardG",
                         "LateralG"])
        for i in range(len(time)):
            writer.writerow([time[i], engine_speed_measure[i],
                             accelerator_opening_angle_measure[i],
                             turn_signal_measure[i], steering_angle_measure[i],
                             speed_estimate[i], brake_oil_pressure_measure[i],
                             yaw_rate_measure[i],
                             forward_and_rearward_g_measure[i],
                             lateral_g_measure[i]])

    plt.plot(speed_measure, label="measurement")
    plt.plot(speed_estimate, label="estimate")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
