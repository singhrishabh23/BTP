import numpy as np

# here what i tried to do is i have specified the number of lasers which is 7 and the step size which is the learning rate for the spgd algo.
N = 7
step_size = 0.01

# here we have random initial phases from 0 to 2*pi for every 7 lasers 
phases = np.random.rand(N) * 2 * np.pi

# the "calculate_error_signal" will calculate the error signal where desired phase is the mean of all the random phases i got 
# and the error signal is the complex exponential of the desired phases and for every random phase
def calculate_error_signal(phases):
    desired_phase = np.mean(phases)
    error_signal = np.exp(1j * (phases - desired_phase))
    return error_signal

# i have taken the gradient of error signal with respect to phase values of each laser 
# i am not very sure of how to calculate the gradient please check this part i have taken the reference from internet sources 
def calculate_gradient(error_signal):
    gradient = np.imag(np.conj(error_signal)[:,np.newaxis] * error_signal[np.newaxis,:]).sum(axis=1)
    return gradient

# now i have calculated the power of each signal by squaring its sin and cosine values 
def calculate_total_power(phases):
    total_power = np.sum(np.cos(phases))*2 + np.sum(np.sin(phases))*2
    return total_power

# here i have set initial total power and iteration count to zero and i will see at what iteration the power stops changing significantly 
total_power = 0
iteration = 0


while True:
    # calculate error signal and gradient
    error_signal = calculate_error_signal(phases)
    gradient = calculate_gradient(error_signal)

    # this expression is updating the phases that reduces the error signal
    phases = phases - step_size * gradient

    # calculate new total power if the new_total_power is reducing this means that the phases are going de-phased
    # else if it is increasing it means it's increasing 
    new_total_power = calculate_total_power(phases)

    # if the convergence has met come out of the loop. the value of 1e-6 is arbitary
    if abs(new_total_power - total_power) < 1e-6:
        break

    # update total power and iteration count
    total_power = new_total_power
    iteration += 1

    # it will give us the value of the number of iterations it took for the algorithm so that 
    # the difference in new and old power is not that significant 
    print(f"Iteration: {iteration}, Total Power: {total_power}")

# there can be different final phase values because of the noise and the randomness of the algo. 
# and also maybe because of its initial conditions 
print("Final phases:")
print(phases)