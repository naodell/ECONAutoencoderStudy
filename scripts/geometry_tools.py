'''
Some tools for visualizing detector elements.
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def hex_to_cartesian(hex_coord, zside=-1):
    # convert hex coordinates to cartesian coords this should change depending
    # on whether you are looking at the +/- z side.  Currently only works of -z
    # side.
    angle = np.pi/6
    hex_radius = 0.95*8*2.54/2
    d = 2*hex_radius*np.cos(angle)
    xcoord = [d*(c[0] - c[1]*np.sin(angle)) for c in hex_coord]
    ycoord = [d*c[1]*np.sin(np.pi/2 - angle) for c in hex_coord]

    # do the above conversion using this at some point
    #trans_matrix = np.array([[1., np.sin(angle)], [0., np.sin(np.pi/2 - angle)]])
    
    return xcoord, ycoord

def draw_module_layers():
    '''
    This plots a single layer of HGCal in the x-y plane.

    To-do:
    ======
     * get actual geometry for each layer
    '''

    # generate coordinates to be used for hexagonal grid
    u = np.arange(-13, 14)
    coord = np.array(list(product(u, u)))
    xcoord, ycoord = hex_to_cartesian(coord)

    # make some plots
    fig, ax = plt.subplots(1, figsize=(16, 16))
    ax.set_aspect('equal')

    # draw hexegonal grid
    inner_radius = 0.82*32.8 # this number comes from the TDR, but is not actually correct
    outer_radius = 160
    for x, y, u, v in zip(xcoord, ycoord, coord.T[0], coord.T[1]):
	
	# filter out wafers
	r = np.sqrt(x**2 + y**2)
	if r < inner_radius or r > outer_radius:
	    continue
	    
	color = 'C0'
	alpha = 0.1
	poly = RegularPolygon((x, y), 
			     numVertices=6, 
			     radius=hex_radius, 
			     orientation=np.radians(0), 
			     facecolor=color, 
			     alpha=alpha, 
			     edgecolor='k',
			     zorder=1
			    )
	ax.add_patch(poly)
	
	# Add text labels
	ax.text(x, y+0.2, f'({u}, {v})', ha='center', va='center', size=10)
	
    # inner hexagon and circle (just for show)
    inner_circle = Circle((0, 0), 
			 radius=inner_radius,
			 facecolor='none', 
			 alpha=0.5, 
			 linestyle='--',
			 edgecolor='r'
			 )
    ax.add_patch(inner_circle)

    poly = RegularPolygon((0, 0), 
			 numVertices=6, 
			 radius=inner_radius/np.cos(angle), 
			 orientation=np.radians(0), 
			 facecolor='none', 
			 alpha=0.5, 
			 linestyle='--',
			 edgecolor='r'
			)
    ax.add_patch(poly)

    # outer hexagon and circle (just for show)
    outer_circle = Circle((0, 0), 
			 radius=160, 
			 facecolor='none', 
			 alpha=0.5, 
			 linestyle='--',
			 edgecolor='r'
			 )
    ax.add_patch(outer_circle)

    poly = RegularPolygon((0, 0), 
			 numVertices=6, 
			 radius=160, 
			 orientation=np.radians(30), 
			 facecolor='none', 
			 alpha=0.5, 
			 edgecolor='r'
			)

    # draw lines every pi/3 radians
    for angle in np.arange(0, 360, 60):
	rad = (angle/180)*np.pi
	plt.plot([0., 200*np.cos(rad)], [0., 200*np.sin(rad)], 'r--', linewidth=0.5, alpha=0.5)
	
    ax.set_xlim(-170, 170)
    ax.set_ylim(-170, 170)
		  
    plt.show()

