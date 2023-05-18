#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilterOneDimension


SAMPLE_DRIVING_DATA_FILE_NAME = "driving_data.csv"
OUTPUT_FILE_NAME = "driving_data_filtered.csv"
OUTPUT_DECIMAL_PLACES = 3

# CSV column names
TIME = "Time"
ENGINE_SPEED = "EngineSpeed"
ACCELERATOR_OPENING_ANGLE = "AcceleratorOpeningAngle"
TURN_SIGNAL = "TurnSignal"
STEERING_ANGLE = "SteeringAngle"
SPEED = "Speed"
BRAKE_OIL_PRESSURE = "BrakeOilPressure"
YAW_RATE = "YawRate"
FORWARD_AND_REARWARD_G = "ForwardAndRearwardG"
LATERAL_G = "LateralG"

COLUMN_NAME_ROW_NUM = 0


def main():
    with open(SAMPLE_DRIVING_DATA_FILE_NAME) as f:
        reader = csv.reader(f)
        data = list(reader)

    # Get the index of each column
    columns_name = data[COLUMN_NAME_ROW_NUM]
    for i in range(len(columns_name)):
        column_name = columns_name[i]
        if TIME in column_name:
            time_idx = i
        elif ENGINE_SPEED in column_name:
            engine_speed_idx = i
        elif ACCELERATOR_OPENING_ANGLE in column_name:
            accelerator_opening_angle_idx = i
        elif TURN_SIGNAL in column_name:
            turn_signal_idx = i
        elif STEERING_ANGLE in column_name:
            steering_angle_idx = i
        elif SPEED in column_name:
            speed_idx = i
        elif BRAKE_OIL_PRESSURE in column_name:
            brake_oil_pressure_idx = i
        elif YAW_RATE in column_name:
            yaw_rate_idx = i
        elif FORWARD_AND_REARWARD_G in column_name:
            forward_and_rearward_g_idx = i
        elif LATERAL_G in column_name:
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

    for i in range(COLUMN_NAME_ROW_NUM + 1, len(data)):
        time.append(float(data[i][time_idx]))
        engine_speed_measure.append(float(data[i][engine_speed_idx]))
        accelerator_opening_angle_measure.append(
            float(data[i][accelerator_opening_angle_idx]))
        turn_signal_measure.append(float(data[i][turn_signal_idx]))
        steering_angle_measure.append(float(data[i][steering_angle_idx]))
        speed_measure.append(float(data[i][speed_idx]))
        brake_oil_pressure_measure.append(
            float(data[i][brake_oil_pressure_idx]))
        yaw_rate_measure.append(float(data[i][yaw_rate_idx]))
        forward_and_rearward_g_measure.append(
            float(data[i][forward_and_rearward_g_idx]))
        lateral_g_measure.append(float(data[i][lateral_g_idx]))

    engine_speed_estimate = []
    accelerator_opening_angle_estimate = []
    steering_angle_estimate = []
    speed_estimate = []
    brake_oil_pressure_estimate = []
    yaw_rate_estimate = []
    forward_and_rearward_g_estimate = []
    lateral_g_estimate = []

    # Set Kalman filter parameters
    # consider the current state as the next state (no change)
    state_transition_matrix = 1.0
    observation_matrix = 1.0  # consider no scaling between the state and the measurement
    process_noise_matrix = 0.05  # guessing value
    measurement_noise = 0.5  # guessing value
    estimated_error = 0.0  # according to the initialization estimate error
    control_matrix = 0.0  # no control input
    measurable_input = 0.0  # no measurable input

    # Initialize Kalman filter
    engine_speed_kf = KalmanFilterOneDimension(engine_speed_measure[0], state_transition_matrix,
                                               observation_matrix, process_noise_matrix,
                                               measurement_noise, estimated_error,
                                               control_matrix, measurable_input)
    accelerator_opening_angle_kf = KalmanFilterOneDimension(accelerator_opening_angle_measure[0], state_transition_matrix,
                                                            observation_matrix, process_noise_matrix,
                                                            measurement_noise, estimated_error,
                                                            control_matrix, measurable_input)
    steering_angle_kf = KalmanFilterOneDimension(steering_angle_measure[0], state_transition_matrix,
                                                 observation_matrix, process_noise_matrix,
                                                 measurement_noise, estimated_error,
                                                 control_matrix, measurable_input)
    speed_kf = KalmanFilterOneDimension(speed_measure[0], state_transition_matrix,
                                        observation_matrix, process_noise_matrix,
                                        measurement_noise, estimated_error,
                                        control_matrix, measurable_input)
    brake_oil_pressure_kf = KalmanFilterOneDimension(brake_oil_pressure_measure[0], state_transition_matrix,
                                                     observation_matrix, process_noise_matrix,
                                                     measurement_noise, estimated_error,
                                                     control_matrix, measurable_input)
    yaw_rate_kf = KalmanFilterOneDimension(yaw_rate_measure[0], state_transition_matrix,
                                           observation_matrix, process_noise_matrix,
                                           measurement_noise, estimated_error,
                                           control_matrix, measurable_input)
    forward_and_rearward_g_kf = KalmanFilterOneDimension(forward_and_rearward_g_measure[0], state_transition_matrix,
                                                         observation_matrix, process_noise_matrix,
                                                         measurement_noise, estimated_error,
                                                         control_matrix, measurable_input)
    lateral_g_kf = KalmanFilterOneDimension(lateral_g_measure[0], state_transition_matrix,
                                            observation_matrix, process_noise_matrix,
                                            measurement_noise, estimated_error,
                                            control_matrix, measurable_input)

    # Execute Kalman filter
    for i in range(len(time)):
        engine_speed_estimate.append(
            engine_speed_kf.execute(engine_speed_measure[i]))
        accelerator_opening_angle_estimate.append(
            accelerator_opening_angle_kf.execute(accelerator_opening_angle_measure[i]))
        steering_angle_estimate.append(
            steering_angle_kf.execute(steering_angle_measure[i]))
        speed_estimate.append(speed_kf.execute(speed_measure[i]))
        brake_oil_pressure_estimate.append(
            brake_oil_pressure_kf.execute(brake_oil_pressure_measure[i]))
        yaw_rate_estimate.append(yaw_rate_kf.execute(yaw_rate_measure[i]))
        forward_and_rearward_g_estimate.append(
            forward_and_rearward_g_kf.execute(forward_and_rearward_g_measure[i]))
        lateral_g_estimate.append(lateral_g_kf.execute(lateral_g_measure[i]))

    # Output the result
    with open(OUTPUT_FILE_NAME, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        for i in range(COLUMN_NAME_ROW_NUM):
            writer.writerow(data[i])
        writer.writerow([TIME, ENGINE_SPEED, ACCELERATOR_OPENING_ANGLE,
                         TURN_SIGNAL, STEERING_ANGLE, SPEED,
                         BRAKE_OIL_PRESSURE, YAW_RATE,
                         FORWARD_AND_REARWARD_G, LATERAL_G])
        for i in range(len(time)):
            writer.writerow([time[i],
                             round(
                                 engine_speed_estimate[i], OUTPUT_DECIMAL_PLACES),
                             round(
                                 accelerator_opening_angle_estimate[i], OUTPUT_DECIMAL_PLACES),
                             turn_signal_measure[i],
                             round(
                                 steering_angle_estimate[i], OUTPUT_DECIMAL_PLACES),
                             round(speed_estimate[i], OUTPUT_DECIMAL_PLACES),
                             round(
                                 brake_oil_pressure_estimate[i], OUTPUT_DECIMAL_PLACES),
                             round(yaw_rate_estimate[i],
                                   OUTPUT_DECIMAL_PLACES),
                             round(
                                 forward_and_rearward_g_estimate[i], OUTPUT_DECIMAL_PLACES),
                             round(lateral_g_estimate[i], OUTPUT_DECIMAL_PLACES)])

    # Plot the result
    fig = plt.figure()
    axes = fig.subplots(nrows=4, ncols=2)
    fig.suptitle("Kalman filter result")
    axes[0, 0].plot(time, engine_speed_measure, label="measure")
    axes[0, 0].plot(time, engine_speed_estimate, label="estimate")
    axes[0, 0].set_title("Engine speed")
    axes[0, 0].set_ylabel("[RPM]")

    axes[0, 1].plot(time, accelerator_opening_angle_measure, label="measure")
    axes[0, 1].plot(time, accelerator_opening_angle_estimate, label="estimate")
    axes[0, 1].set_title("Accelerator opening angle")
    axes[0, 1].set_ylabel("[%]")

    axes[1, 0].plot(time, steering_angle_measure, label="measure")
    axes[1, 0].plot(time, steering_angle_estimate, label="estimate")
    axes[1, 0].set_title("Steering angle")
    axes[1, 0].set_ylabel("[°]")

    axes[1, 1].plot(time, speed_measure, label="measure")
    axes[1, 1].plot(time, speed_estimate, label="estimate")
    axes[1, 1].set_title("Speed")
    axes[1, 1].set_ylabel("[km/h]")

    axes[2, 0].plot(time, brake_oil_pressure_measure, label="measure")
    axes[2, 0].plot(time, brake_oil_pressure_estimate, label="estimate")
    axes[2, 0].set_title("Brake oil pressure")
    axes[2, 0].set_ylabel("[MPa]")

    axes[2, 1].plot(time, yaw_rate_measure, label="measure")
    axes[2, 1].plot(time, yaw_rate_estimate, label="estimate")
    axes[2, 1].set_title("Yaw rate")
    axes[2, 1].set_ylabel("[°/s]")

    axes[3, 0].plot(time, forward_and_rearward_g_measure, label="measure")
    axes[3, 0].plot(time, forward_and_rearward_g_estimate, label="estimate")
    axes[3, 0].set_title("Forward and rearward G")
    axes[3, 0].set_ylabel("[m/s^2]")

    axes[3, 1].plot(time, lateral_g_measure, label="measure")
    axes[3, 1].plot(time, lateral_g_estimate, label="estimate")
    axes[3, 1].set_title("Lateral G")
    axes[3, 1].set_ylabel("[m/s^2]")

    fig.tight_layout()

    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc="upper right")

    # Show the result
    plt.show()


if __name__ == "__main__":
    main()
