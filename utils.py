# Function definitions for the project
import numpy as np
import matplotlib.pyplot as plt
def damped_traub(f, df, z, tol = 1e-15, delta = 1, max_iter = 100):
    """
    Damped Newton's method with Traub's modification
    :param f: Function to find root of
    :param df: Derivative of f
    :param z: Initial guess
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :return: None if no convergence, otherwise root and number of iterations
    """
    # Maximum 50 iterations
    for i in range(max_iter):
        if abs(df(z)) > tol:
            newt_step = z - f(z)/df(z)
            z_new = newt_step - delta * f(newt_step)/df(z)
        else:
            return None, max_iter # Return None if derivative is too small

        # Stop the method when two iterates are close or f(z) = 0
        if abs(f(z)) < tol or abs(z_new - z) < tol:
            return z_new, i
        else:
            # Update z and continue
            z = z_new
    # If no convergence, return None
    return None, max_iter

def halley(f, df, d2f, z, tol = 1e-15, delta = 1, max_iter = 100):
    """
    Damped Newton's method with Traub's modification
    :param f: Function to find root of
    :param df: Derivative of f
    :param d2f: Second derivative of f
    :param z: Initial guess
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :return: None if no convergence, otherwise root and number of iterations
    """
    # Maximum 50 iterations
    for i in range(max_iter):
        if abs(df(z)) > tol:
            z_new = z- 2*f(z)*df(z)/(2*df(z)**2 - f(z)*d2f(z))
        else:
            return None, max_iter # Return None if derivative is too small

        # Stop the method when two iterates are close or f(z) = 0
        if abs(f(z)) < tol or abs(z_new - z) < tol:
            return z_new, i
        else:
            # Update z and continue
            z = z_new
    # If no convergence, return None
    return None, max_iter
def plot_damped_traub(f, df, tol = 1e-15, delta = 1, N = 2000, xmin = -1, xmax = 1, ymin = -1, ymax = 1, max_iter = 100):
    """
    Plots the convergence of damped Traub's method
    :param f: Function to find root of
    :param df: Derivative of f
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param N: Number of points to plot
    :param xmin: Minimum x value for plot
    :param xmax: Maximum x value for plot
    :param ymin: Minimum y value for plot
    :param ymax: Maximum y value for plot
    :param max_iter: Maximum number of iterations
    :return: None
    """
    # List to store unique roots
    roots = []

    # Define the ranges for z_x and z_y
    z_x_range = np.linspace(xmin, xmax, N)
    z_y_range = np.linspace(ymin, ymax, N)

    # Create a meshgrid from the ranges
    z_x, z_y = np.meshgrid(z_x_range, z_y_range)

    # Create an array to store the number of iterations
    iterations_array = np.zeros_like(z_x)

    # Iterate over the meshgrid
    for i in range(N):
        for j in range(N):

            # Create a complex number from the meshgrid
            point = complex(z_x[i,j], z_y[i,j])

            # Apply damped Traub's method
            root, iterations = damped_traub(f, df, point, tol, delta, max_iter)

            # Store the number of iterations
            iterations_array[i,j] = iterations

            # Check if the root is found
            if root:
                flag = False
                # Check if the root is already in the list
                for test_root in roots:
                    if abs(test_root - root) < tol*1e7:
                        root = test_root
                        flag = True
                        break
                # If the root is not in the list, append it
                if not flag:
                    roots.append(root)

    # Define the maximum number of iterations for normalization
    max_iterations = np.max(iterations_array)
    min_iterations = np.min(iterations_array)
    # Plot the colored picture
    plt.figure(figsize=(10,10))
    plt.imshow(iterations_array, extent = [xmin, xmax, ymin, ymax], cmap = 'hsv', vmax = max_iterations, vmin = min_iterations, origin='lower')

    # Plot the roots
    root_markers = np.array(roots)
    plt.scatter(root_markers.real, root_markers.imag, marker = 'o', color = 'black', s = 20)

    # Remove the axes
    plt.axis('off')

    # Show the plot
    plt.show()
# --------------------------------------------------------------------------------------------
# Function for Damped Traub's method colored plot
def darken(color, fraction):
    """
    Darkens a color by a fraction
    :param color: Color to darken
    :param fraction: Fraction to darken by
    :return: Darkened color
    """
    return [p * (1 - fraction) for p in color]

def normalize(bounds, perc):
    """
    Normalizes the bounds by a percentage
    :param bounds: Bounds to normalize
    :param perc: Percentage to normalize by
    :return: Normalized bounds
    """
    a = bounds[0]
    b = bounds[1]

    return (b-a) * perc + a

def pixel_color(x,y, bounds_x, bounds_y, width, height, f, df, roots, colors,tol, delta, max_iter):
    """
    Returns the color of a pixel
    :param x: x-coordinate of the pixel
    :param y: y-coordinate of the pixel
    :param bounds_x: Bounds for the x-axis
    :param bounds_y: Bounds for the y-axis
    :param width: Width of the plot
    :param height: Height of the plot
    :param f: Function to find root of
    :param df: Derivative of f
    :param roots: Roots of the function
    :param colors: Colors of the roots
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :return: Color of the pixel
    """
    real = normalize(bounds_x, x/ width)
    imag = normalize(bounds_y, y/ height)
    return point_color(complex(real, imag), f, df, roots, colors, delta=delta, tol=tol, max_iter=max_iter)

def point_color(z, f, df, roots, colors, tol = 1e-15, delta = 1, max_iter = 100):
    """
    Returns the color of a point
    :param z: Point to find the color of
    :param f: Function to find root of
    :param df: Derivative of f
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :param roots: Roots of the function
    :param colors: Colors of the roots
    :return: Color of the point
    """
    # Perform the damped Traub's method assigning color intensity to the point
    for i in range(max_iter):
        if abs(df(z)) < tol:
            return [0,0,0]
        else:
            newt_step = z - f(z)/df(z)
            z_new = newt_step - delta * f(newt_step)/df(z)
            for root_id, root in enumerate(roots):
                diff = abs(z_new - root)
                if diff > tol:
                    z = z_new
                    continue
                # Found which attractor the point converges to
                color_intensity = max(min(i / (1 << 5), 0.95), 0)
                return darken(colors[root_id], color_intensity)
    return [0,0,0]
def pixel_color_halley(x,y, bounds_x, bounds_y, width, height, f, df,d2f, roots, colors,tol, max_iter):
    """
    Returns the color of a pixel
    :param x: x-coordinate of the pixel
    :param y: y-coordinate of the pixel
    :param bounds_x: Bounds for the x-axis
    :param bounds_y: Bounds for the y-axis
    :param width: Width of the plot
    :param height: Height of the plot
    :param f: Function to find root of
    :param df: Derivative of f
    :param d2f: Second derivative of f
    :param roots: Roots of the function
    :param colors: Colors of the roots
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :return: Color of the pixel
    """
    real = normalize(bounds_x, x/ width)
    imag = normalize(bounds_y, y/ height)
    return point_color_halley(complex(real, imag), f, df,d2f, roots, colors, tol=tol, max_iter=max_iter)

def point_color_halley(z, f, df,d2f, roots, colors, tol = 1e-15, max_iter = 100):
    """
    Returns the color of a point
    :param z: Point to find the color of
    :param f: Function to find root of
    :param df: Derivative of f
    :param d2f: Second derivative of f
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :param roots: Roots of the function
    :param colors: Colors of the roots
    :return: Color of the point
    """
    # Perform the damped Traub's method assigning color intensity to the point
    for i in range(max_iter):
        if abs(df(z)) < tol:
            return [0,0,0]
        else:
            z_new = z - 2*f(z)*df(z)/(2*df(z)**2 - f(z)*d2f(z))
            for root_id, root in enumerate(roots):
                diff = abs(z_new - root)
                if diff > tol:
                    z = z_new
                    continue
                # Found which attractor the point converges to
                color_intensity = max(min(i / (1 << 5), 0.95), 0)
                return darken(colors[root_id], color_intensity)
    return [0,0,0]
def plot_colored_halley(f, df, d2f, bounds_x, bounds_y, width, height, roots, colors, tol = 1e-15, max_iter = 100):
    """
    Plots the convergence of damped Traub's method
    :param f: Function to find root of
    :param df: Derivative of f
    :param d2f: Second derivative of f
    :param bounds_x: Bounds for the x-axis
    :param bounds_y: Bounds for the y-axis
    :param width: Width of the plot
    :param height: Height of the plot
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :param roots: Roots of the function
    :param colors: Colors of the roots
    :return: None
    """
    data = np.zeros((height, width, 3), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            # Assign the color to the pixel
            data[y, x] = pixel_color_halley(x, y, bounds_x, bounds_y, width, height, f, df,d2f, roots, colors, tol=tol, max_iter=max_iter)

    # Plot the colored picture
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(data, extent = [bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]], origin='lower')
    # Plot the roots
    ax.scatter([root.real for root in roots], [root.imag for root in roots], marker = 'o', color = 'black', s = 20)
    # Plotting initial conditions Hubbard et al.
    '''
    r = 2.283
    N = 67
    circle = np.zeros(N, dtype=complex)
    for i in range(N):
        theta = 2*np.pi*i/N
        circle[i] = r*np.exp(1j*theta)
    ax.scatter([c.real for c in circle], [c.imag for c in circle], marker = 'o', color = 'white', s = 20)
    '''
    plt.axis('off')
    plt.show()
def plot_colored_damped_traub(f, df, bounds_x, bounds_y, width, height, roots, colors, tol = 1e-15, delta = 1, max_iter = 100):
    """
    Plots the convergence of damped Traub's method
    :param f: Function to find root of
    :param df: Derivative of f
    :param bounds_x: Bounds for the x-axis
    :param bounds_y: Bounds for the y-axis
    :param width: Width of the plot
    :param height: Height of the plot
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :param roots: Roots of the function
    :param colors: Colors of the roots
    :return: None
    """
    data = np.zeros((height, width, 3), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            # Assign the color to the pixel
            data[y, x] = pixel_color(x, y, bounds_x, bounds_y, width, height, f, df, roots, colors, delta= delta, tol=tol, max_iter=max_iter)

    # Plot the colored picture
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(data, extent = [bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]], origin='lower')
    # Plot the roots
    ax.scatter([root.real for root in roots], [root.imag for root in roots], marker = 'o', color = 'black', s = 20)
    # Plotting initial conditions Hubbard et al.
    '''
    r = 2.283
    N = 67
    circle = np.zeros(N, dtype=complex)
    for i in range(N):
        theta = 2*np.pi*i/N
        circle[i] = r*np.exp(1j*theta)
    ax.scatter([c.real for c in circle], [c.imag for c in circle], marker = 'o', color = 'white', s = 20)
    '''
    plt.axis('off')
    plt.show()

# --------------------------------------------------------------------------------------------
# Parameter plane for the quadratic case
def iterate_G(z, delta, max_iter=100, tol=1e-8):
    """
    Iteratively applies a function G(z) to see if converge to 0, infinity or neither
    :param z: Initial guess
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :param tol: Tolerance for convergence
    :return: Number of iterations if convergence is reached, otherwise max_iter
    """
    for i in range(max_iter):
        z_new = (z*z)*((z*z + 2*z + (1-delta))/((1-delta)*z*z + 2*z + 1))
        # If convergence is reached to one of the two superattracting fixed points, return the number of iterations
        if abs(z_new - 0) < tol or abs(z_new)>1e6:
            return i
        else:
            z = z_new
    return max_iter

def plot_parameter_plane(xmin=-4.5, xmax=6.5, ymin=-6, ymax=6, N=3000, max_iter=100, tol=1e-8):
    """
    Plots the parameter plane for the iterate_G function
    :param xmin: Minimum value of the real part of delta
    :param xmax: Maximum value of the real part of delta
    :param ymin: Minimum value of the imaginary part of delta
    :param ymax: Maximum value of the imaginary part of delta
    :param N: Number of points along each dimension for the meshgrid
    :param max_iter: Maximum number of iterations for iterate_G function
    :param tol: Tolerance for convergence in iterate_G function
    :return: None
    """

    # Define the ranges for delta_x and delta_y
    delta_x_range = np.linspace(xmin, xmax, N)
    delta_y_range = np.linspace(ymin, ymax, N)

    # Create a meshgrid from the ranges
    delta_x, delta_y = np.meshgrid(delta_x_range, delta_y_range)

    # Create an array to store the number of iterations
    iterations_array = np.zeros_like(delta_x)

    # Iterate over the meshgrid
    for i in range(N):
        for j in range(N):

            # Create a complex number from the meshgrid
            delta = complex(delta_x[i,j], delta_y[i,j])

            # Iterate G over the critical point
            crit_point = (-(2+delta) + np.sqrt((2+delta)**2 - 4*(1-delta)**2))/(2*(1-delta))
            iterations = iterate_G(crit_point, delta, max_iter, tol)

            # Store the number of iterations
            iterations_array[i,j] = iterations

    # Define the maximum number of iterations for normalization
    max_iterations = np.max(iterations_array)
    min_iterations = np.min(iterations_array)

    # Plot the parameter plane
    plt.figure(figsize=(10,10))
    plt.imshow(iterations_array, extent = [xmin, xmax, ymin, ymax], cmap = 'hsv', vmax = max_iterations, vmin = min_iterations, origin='lower')

    # Plot delta=0 and delta=1
    plt.scatter(0, 0, marker = 'o', color = 'black', s = 20)
    plt.scatter(1, 0, marker = 'o', color = 'black', s = 20)

    # Annotate the scatter points with text
    plt.text(0, 0, '0', fontsize=16, ha='right', va='bottom')  # Text '0' for (0,0)
    plt.text(1, 0, '1', fontsize=16, ha='right', va='bottom')  # Text '1' for (1,0)


    # Remove the axes
    plt.axis('off')

    # Show the plot
    plt.show()