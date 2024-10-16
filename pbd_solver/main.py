from .vine import *


def vis_init():
    plt.figure(1, figsize=(10, 10))
    
    global main_ax, fig_ax
    main_ax = plt.gca()
    main_ax.set_aspect('equal')
    main_ax.set_xlim(-ww, ww)
    main_ax.set_ylim(-ww, ww)
    
    plt.figure(2, figsize=(10, 5))
    fig_ax = plt.gca()

def draw(vine: Vine):
    global main_ax, fig_ax
    main_ax.cla()
    
    main_ax.set_aspect('equal')
    main_ax.set_xlim(-ww, ww)
    main_ax.set_ylim(-ww, ww)
    
    # Draw the obstacles
    for obstacle in vine.obstacles:
        obstacle_patch = Rectangle((obstacle[0], obstacle[1]),
                                    obstacle[2] - obstacle[0], obstacle[3] - obstacle[1],
                                    linewidth=1, edgecolor='r', facecolor='gray')
        main_ax.add_patch(obstacle_patch)

    # Draw each body
    for i in range(vine.nbodies - 1):
        # Body endpoints
        x_start = vine.xslice[i] - vine.d[i] * torch.cos(vine.thetaslice[i])
        y_start = vine.yslice[i] - vine.d[i] * torch.sin(vine.thetaslice[i])
        x_end = vine.xslice[i] + vine.d[i] * torch.cos(vine.thetaslice[i])
        y_end = vine.yslice[i] + vine.d[i] * torch.sin(vine.thetaslice[i])

        main_ax.plot([x_start, x_end], [y_start, y_end], c='blue', linewidth=10)
        # main_ax.scatter(vine.xslice[i], vine.yslice[i], c='pink', s=radius2pt(vine.radius))       
    
    # Draw last body
    x_start = vine.xslice[-2] + vine.d[-1] * torch.cos(vine.thetaslice[-2])
    y_start = vine.yslice[-2] + vine.d[-1] * torch.sin(vine.thetaslice[-2])
    x_end = vine.xslice[-1]
    y_end = vine.yslice[-1]
    main_ax.plot([x_start, x_end], [y_start, y_end], c='blue', linewidth=10)
        
    for x, y in zip(vine.xslice, vine.yslice):
        circle = plt.Circle((x, y), vine.radius, color='g', fill=False)
        main_ax.add_patch(circle)
    
    if hasattr(vine, 'dbg_dist'):
        for x, y, dist, contact in zip(vine.xslice, vine.yslice, vine.dbg_dist, vine.dbg_contactpts):
            # Distance text
            # main_ax.text(x + 1, y + 0, f'{dist:.3f}')
            # Contact point
            # main_ax.arrow(x, y, contact[0] - x, contact[1] - y)
            pass 
        
if __name__ == '__main__':
    vine = Vine(nbodies=4, init_heading_deg=-45, obstacles = [
            (40, -70, 55, -10) # x, y, x2, y2
        ], grow_rate=20)
    
    vis_init()
    
    draw(vine)
    plt.pause(0.001)
    
    for frame in range(1000):
        vine.evolve()
        
        draw(vine)
        
        if frame % 1 == 0:
            plt.pause(0.001)
        
    plt.show()