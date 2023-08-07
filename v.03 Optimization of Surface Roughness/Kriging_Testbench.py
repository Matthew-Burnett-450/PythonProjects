import numpy as np
from noise import snoise2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from Ok_Uk_Module import *
import concurrent.futures

# Set the dimensions of the terrain
width = 500
height = 500

# Set the scale of the terrain features
scale = 0.025

import numpy as np
from noise import snoise2
from scipy.ndimage import gaussian_filter

#Simulated terrain test
def generate_terrain(width, height, scale, octaves, persistence, sigma,z):
    # Initialize the terrain
    terrain = np.zeros((width, height))

    # Generate the fractal noise
    for i in range(width):
        for j in range(height):
            frequency = scale
            amplitude = 1.0
            for _ in range(octaves):
                terrain[i][j] += snoise2(i * frequency, j * frequency) * amplitude
                frequency *= 2  # Double the frequency at each octave
                amplitude *= persistence  # Reduce the amplitude by the persistence at each octave

    # Apply Gaussian blur
    terrain = gaussian_filter(terrain, sigma=sigma)

    return np.abs(terrain)*z


def sample_points(terrain, num_points):
    # Get the width and height of the terrain
    width, height = terrain.shape

    # Generate random x and y coordinates
    x_coords = np.random.randint(0, width, num_points)
    y_coords = np.random.randint(0, height, num_points)

    # Sample the z values from the terrain
    z_values = terrain[x_coords, y_coords]

    # Return the coordinates and z values as numpy arrays
    return np.array([x_coords, y_coords]).T, z_values


def run_test(disp=False):
    terrain = generate_terrain(width=50, height=50, scale=np.random.uniform(low=.01,high=.02), octaves=np.random.randint(low=15,high=25), persistence=.1, sigma=2,z=10)
    points,zpoints=sample_points(terrain=terrain,num_points=75)    
    try:
        krige=UniversalKriging(points,zpoints,Variogram='exponential',trendfunc='cubic')
        zmap=krige.AutoKrige(step=1,xmax=50,ymax=50,ymin=0,xmin=0)
        mse=np.mean((terrain-zmap.T)**2)
        print(mse)

    except:
        np.save('terrain_map_of_error',terrain)
        np.savetxt('samplepoints.csv',np.column_stack(points,zpoints))
    if disp == True:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        vmin = np.min([zmap, terrain])
        vmax = np.max([zmap, terrain])
        
        # Display the first terrain in the first subplot
        im1=ax1.imshow(zmap, cmap='terrain', vmin=vmin, vmax=vmax,origin='lower')
        ax1.set_title('Krige Guess')

        # Display the second terrain in the second subplot
        im2 = ax2.imshow(terrain.T, cmap='terrain',vmin=vmin, vmax=vmax,origin='lower')
        ax2.scatter(points[:,0],points[:,1],facecolors='none', edgecolors='r',s=1)
        ax2.set_title('Terrain')
        fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal')
        # Show the figure
        
        plt.show()
    

#Terrain test

n=20

t0 = time.time()

for i in range(n):
    t_0 = time.time()
    run_test(disp=True)
    t_1 = time.time()
    print(t_1-t_0)

t1 = time.time()
totalexetime = t1-t0

print(totalexetime)
print(f'avg time _ {totalexetime/n}')