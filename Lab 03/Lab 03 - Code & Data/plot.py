import matplotlib.pyplot as plt
import numpy as np

def plot_reprojection_error(rp_error, point_ids, img_id):
   
    maxx = abs(rp_error).max()
    scale = maxx/16 # empirical factor
    fig, ax = plt.subplots(figsize=(6,6))
    for i,p in enumerate(rp_error):
        ax.arrow(0, 0, p[0], p[1], width=0.001*scale, color="orangered", 
                 head_width=0.5*scale, head_length=0.5*scale, overhang=0.5*scale, ec ='orangered')
        ax.text( p[0], p[1], int(point_ids[i]), color ="orangered" )
    
    if maxx > 1:
        ax.text(maxx+0.5*scale,0,'$\epsilon_u$ [pix]', color='dimgray')
        ax.text(0, maxx+0.5*scale,'$\epsilon_v$ [pix]', color='dimgray')
    else:
        ax.text(maxx+0.5*scale,0,'$\epsilon_u$ [-]', color='dimgray')
        ax.text(0, maxx+0.5*scale,'$\epsilon_v$ [-]', color='dimgray')        

    ax.spines['top'].set_color('none') 
    ax.spines['right'].set_color('none') 
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', colors='dimgray')  
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['bottom'].set_color('dimgray') 
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', colors='dimgray') 
    ax.spines['left'].set_position(('data',0))
    ax.spines['left'].set_color('dimgray') 
    
    ax.set_xlim(-maxx-0.5*scale,maxx+0.5*scale)
    ax.set_ylim(-maxx-0.5*scale,maxx+0.5*scale)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Re-projection error of GCPs - Image %s" % img_id)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

def plot_reprojection_error_norm(epsilon, point_ids, img_id):
    maxx = abs(epsilon).max()
    fig, ax = plt.subplots(figsize=(10,6))
    for i,p in enumerate(epsilon):
        ax.scatter( i+1, p, color="orangered")
        ax.text(i+1, p, int(point_ids[i]), color ="orangered")
    ax.plot([0,12.5],[np.mean(epsilon),np.mean(epsilon)])
    
    ax.set_xlabel('GCP')
    if maxx > 1:
        ax.set_ylabel('$\epsilon_r$ [pix]')
    else:
        ax.set_ylabel('$\epsilon_r$ [-]')  
    plt.title("Norm of re-projection error of GCPs - Image %s" % img_id)
    plt.grid(True, linestyle=':')
    plt.show()
    
