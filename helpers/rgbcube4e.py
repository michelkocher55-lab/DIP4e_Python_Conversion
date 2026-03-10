import numpy as np
import matplotlib.pyplot as plt

def rgbcube4e(vx=10, vy=10, vz=4):
    """
    Displays an RGB cube.
    
    fig = rgbcube4e(vx, vy, vz)
    
    Parameters
    ----------
    vx, vy, vz : float, optional
        Viewpoint coordinates (Cartesian). Default is (10, 10, 4).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object showing the cube.
    """
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Grid density for smooth interpolation
    N = 32
    # Matplotlib's plot_surface uses one color per face. To fake a gradient/interpolation,
    # we need a fine mesh (e.g. N=32) so that each small quad varies slightly in color.
    
    u = np.linspace(0, 1, N)
    v = np.linspace(0, 1, N)
    U, V = np.meshgrid(u, v)
    
    # Define 6 faces
    faces = []
    
    # Face 1: Z=0 (Black-Red-Green-Yellow) -> Bottom
    # X varying, Y varying, Z=0
    faces.append((U, V, np.zeros_like(U)))
    
    # Face 2: Z=1 (Blue-Magenta-Cyan-White) -> Top
    faces.append((U, V, np.ones_like(U)))
    
    # Face 3: Y=0 (Black-Red-Blue-Magenta) -> Front (if we consider standard axes)
    # X varying, Z varying, Y=0
    faces.append((U, np.zeros_like(U), V))
    
    # Face 4: Y=1 (Green-Yellow-Cyan-White) -> Back
    faces.append((U, np.ones_like(U), V))
    
    # Face 5: X=0 (Black-Green-Blue-Cyan) -> Left
    # Y varying, Z varying, X=0
    faces.append((np.zeros_like(U), U, V))
    
    # Face 6: X=1 (Red-Yellow-Magenta-White) -> Right
    faces.append((np.ones_like(U), U, V))
    
    # Plot each face
    for X, Y, Z in faces:
        # Create RGB color for each point
        # R = X, G = Y, B = Z
        # stack to (N, N, 3)
        C = np.dstack((X, Y, Z))
        
        ax.plot_surface(X, Y, Z, facecolors=C, shade=False)

    # Setup Axes
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('RGB Color Cube')
    
    # Labels for corners (simplified)
    corners = [
        (0,0,0, 'Black'), (1,0,0, 'Red'), (0,1,0, 'Green'), (0,0,1, 'Blue'),
        (1,1,0, 'Yellow'), (1,0,1, 'Magenta'), (0,1,1, 'Cyan'), (1,1,1, 'White')
    ]
    for x, y, z, label in corners:
        ax.scatter(x, y, z, color='k', s=20)
        # Offset label slightly
        ax.text(x, y, z, label, fontsize=9)

    # Viewpoint Logic
    # Convert Cartesian (vx, vy, vz) to Elev/Azim
    # r = sqrt(vx^2 + vy^2 + vz^2)
    # azim = arctan2(vy, vx) (degrees)
    # elev = arcsin(vz / r) (degrees)
    r = np.sqrt(vx**2 + vy**2 + vz**2)
    azim = np.degrees(np.arctan2(vy, vx))
    if r > 0:
        elev = np.degrees(np.arcsin(vz / r))
    else:
        elev = 0 # Default if view is (0,0,0) - unlikely
        
    # Correct Azimuth convention differences?
    # Matplotlib Azim=0 is X axis? CHECK.
    # Actually Matplotlib default: -60, 30.
    # Usually: azim=0 -> View from -Y axis? No.
    # Standard spherical: azim is angle in x-y plane.
    # Let's trust the math: arctan2(y, x).
    
    # But wait, standard matplotlib view_init(elev, azim) documentation:
    # "azim stores the azimuth angle in the x,y plane. A value of 0 means viewing from the +x direction?? No"
    # Testing usually required. But conversion formula is generally robust.
    # MATLAB view definition: azimuth is rotation around z-axis from -y axis.
    # arctan2(y, x) gives angle from +x axis.
    # MATLAB view(az, el) usage also exists.
    # But this function takes (vx, vy, vz).
    # MATLAB: [az, el] = view(vx, vy, vz)
    
    # We'll use the calculated one.
    ax.view_init(elev=elev, azim=azim)
    
    return fig
