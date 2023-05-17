class KalmanFilterOneDimension:
    # https://www.kalmanfilter.net/multiSummary.html
    def __init__(self, initial_value, state_transition_matrix=1.0,
                 observation_matrix=1.0, process_noise_matrix=0.05,
                 measurement_noise=0.1, estimated_error=0.1,
                 control_matrix=0.0, measurable_input=0.0):
        self.F = state_transition_matrix  # state transition matrix
        self.H = observation_matrix  # observation matrix
        # variance state transition matrix (process noise matrix)
        self.Q = process_noise_matrix
        # measurement noise covariance matrix (variance observation matrix)
        self.R = measurement_noise
        # squared uncertainty of an estimate (covariance matrix)
        self.P = estimated_error ** 2
        self.G = control_matrix  # control matrix
        self.U = measurable_input  # measurable (deterministic) input
        # Skip process noise, i.e. an unmeasurable input that affects the state because it does not typically appear directly in the equations of interest
        # Instead, this term is used to model the uncertainty in the covariance extrapolation equation
        self.kalman_gain = 0.0
        self.predicted_value = 0.0
        self.estimated_value = initial_value

    def __predict(self):
        # https://www.kalmanfilter.net/stateextrap.html
        self.predicted_value = self.F * self.estimated_value + self.G * self.U
        # https://www.kalmanfilter.net/covextrap.html
        self.P = self.F * self.P * self.F + self.Q

    def __update(self, measurement):
        # https://www.kalmanfilter.net/kalmanGain.html
        self.kalman_gain = self.P * self.H / \
            (self.H * self.P * self.H + self.R)
        # https://www.kalmanfilter.net/stateUpdate.html
        self.estimated_value = self.predicted_value + self.kalman_gain * \
            (measurement - self.H * self.predicted_value)
        # https://www.kalmanfilter.net/simpCovUpdate.html
        self.P = (1 - self.kalman_gain * self.H) * self.P

    def execute(self, measurement):
        self.__predict()
        self.__update(measurement)
        return self.estimated_value
