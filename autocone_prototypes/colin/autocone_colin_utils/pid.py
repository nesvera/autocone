import time

class PID:

    def __init__(self, kp, ki, kd, windup):

        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        
        self.sample_time = 0
        self.last_time = time.time()

        self.clear()

        self.windup_guard = windup

    def clear(self):
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, target, feedback_value):
        error = target - feedback_value

        current_time = time.time()
        delta_time = current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.p_term = self.Kp * error
            self.i_term += error * delta_time

            if (self.i_term < -self.windup_guard):
                self.i_term = -self.windup_guard
            elif (self.i_term > self.windup_guard):
                self.i_term = self.windup_guard

            self.d_term = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = current_time
            self.last_error = error

            self.output = self.p_term + (self.Ki * self.i_term) + (self.Kd * self.d_term)

        return self.output