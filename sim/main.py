from .vine import *
from .render import *

if __name__ == '__main__':
    
    ipm = 39.3701/600 # inches per mm
    b1 = [5.5/ipm,-5/ipm,4/ipm,7/ipm]
    b2 = [4/ipm,-17/ipm,7/ipm,5/ipm]
    b3 = [13.5/ipm,-17/ipm,8/ipm,11/ipm]
    b4 = [13.5/ipm,-30/ipm,4/ipm,5/ipm]
    b5 = [20.5/ipm,-30/ipm,4/ipm,5/ipm]
    b6 = [13.5/ipm,-30/ipm,10/ipm,1/ipm]
    
    obstacles = [
            b1, b2, b3, b4, b5, b6
        ]
    
    for i in range(len(obstacles)):
        obstacles[i][0] -= 20
        obstacles[i][1] -= 0
        
        obstacles[i][2] = obstacles[i][0] + obstacles[i][2]
        obstacles[i][3] = obstacles[i][1] + obstacles[i][3]
        
    params = VineParams(nbodies=2, init_heading_deg=-45, obstacles=obstacles, grow_rate=250) 
    
    state, dstate = create_state(params)
    
    vis_init()
    draw(params, state, dstate)
    plt.pause(0.001)
    # plt.show()
    
    for frame in range(1000):
        state, dstate = forward(params, state, dstate)
        
        draw(params, state, dstate)
        
        if frame % 1 == 0:
            plt.pause(0.001)
        
    plt.show()