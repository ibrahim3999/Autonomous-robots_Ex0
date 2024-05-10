import numpy as np
import pandas as pd


def read_gnss_data(filepath):
    """Read GNSS data from a CSV file."""
    return pd.read_csv(filepath)


def compute_initial_guess(sat_positions):
    """Compute an initial guess for the receiver's position as the centroid of satellite positions."""
    return np.mean(sat_positions, axis=0)


def compute_pseudoranges(receiver_pos, sat_positions):
    """Compute the geometric distances between receiver and satellites."""
    return np.linalg.norm(sat_positions - receiver_pos, axis=1)


def update_position_estimate(receiver_pos, sat_positions, measured_pseudoranges, weights):
    """Update receiver position using a weighted least squares estimate."""
    design_matrix = np.hstack((sat_positions - receiver_pos, np.ones((sat_positions.shape[0], 1))))
    pseudorange_residuals = measured_pseudoranges - compute_pseudoranges(receiver_pos, sat_positions)
    pseudorange_residuals = pseudorange_residuals[:, np.newaxis]  # Convert to column vector
    weights_matrix = np.diag(weights)
    try:
        correction = np.linalg.inv(
            design_matrix.T @ weights_matrix @ design_matrix) @ design_matrix.T @ weights_matrix @ pseudorange_residuals
    except np.linalg.LinAlgError:
        return receiver_pos  # Return current position if matrix is singular/non-invertible
    return receiver_pos + correction[:-1].flatten(), correction[-1].flatten()  # Position update and clock bias update


def iterative_least_squares(data, initial_pos, max_iterations=10, convergence_threshold=1e-4):
    receiver_pos = initial_pos
    for _ in range(max_iterations):
        sat_positions = data[['Sat.X', 'Sat.Y', 'Sat.Z']].values
        measured_pseudoranges = data['Pseudo-Range'].values
        weights = 1 / (data[
                           'CN0'].values ** 2)  # Weight by the square of the CN0 values, assuming higher CN0 means lower noise

        new_pos, clock_bias = update_position_estimate(receiver_pos, sat_positions, measured_pseudoranges, weights)
        if np.linalg.norm(new_pos - receiver_pos) < convergence_threshold:
            break
        receiver_pos = new_pos

    return receiver_pos, clock_bias


# Example Usage
if __name__ == '__main__':
    gnss_data = read_gnss_data('gnss_measurements_output.csv')
    initial_pos = compute_initial_guess(gnss_data[['Sat.X', 'Sat.Y', 'Sat.Z']].values)
    final_position, clock_bias = iterative_least_squares(gnss_data, initial_pos)
    print("Final Receiver Position:", final_position)
    print("Estimated Clock Bias:", clock_bias)