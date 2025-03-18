from scipy.interpolate import CubicSpline, interp1d
import numpy as np
import matplotlib.pyplot as plt

def lens_eq(Psi,x1,x2):
    '''lens eq'''
    y1 = x1 - Psi.dpsi_dx1(x1,x2)
    y2 = x2 - Psi.dpsi_dx2(x1,x2)
    
    return y1,y2

def plot_singular_curves(ax, psi, mesh = (None, None), target_range = 10, CausticInterpolation = True, pointsIncrease = 5, method = 'spline'):
    """
    Plots the caustics and critical curves of the glow lens defined by `psi` onto the given axis (`ax`).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axis on which to plot the curves.
    psi : glow lens
        The lens potential function.
    mesh : tuple of np.ndarray, optional
        A 2D NumPy meshgrid used to compute and plot the critical curves. If not provided, 
        the function will automatically generate a mesh within the range (-target_range, target_range) in both x and y.
    target_range : float, optional
        The (half) range within which the function will search for critical curves if `mesh` is not provided (default is 10).
    CausticInterpolation : bool, optional
        If True, interpolation is applied to the critical curves before computing the caustics, improving smoothness (default is True).
    pointsIncrease : int, optional
        The factor by which the initial number of sampled points is increased when interpolation is enabled (default is 5).
    method : {'spline', 'linear'}, optional
        The interpolation method applied when `CausticInterpolation` is True. If 'spline', splines are used; 
        otherwise, linear interpolation is applied (default is 'spline').
         
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plotted critical curves and caustics.

    Notes
    -----
    - Caustics are computed from the critical curves via the lens equation.
    - If the produced curves appear visually incomplete, try specifying a more suitable `target_range` or providing a custom `mesh`.
    - In some cases, closely spaced but distinct critical curves may lead to spurious results during interpolation. 
      Changing the interpolation method or disabling `CausticInterpolation` can help verify the accuracy of the results.

    Example
    -----
      The following example demonstrates a specific case of lens creation and use of the function, in which setting dx=0.45 will show spurious results:
  
      ```python
        from glow import lenses
        
        dx = 0.5
        psi0s = [1, 0.1]
        create_sublens = lambda psi0, x1, x2: lenses.Psi_offcenterSIS({'psi0':psi0, 'xc1':x1, 'xc2':x2})
        sublenses = [create_sublens(psi0, xlens,0) for psi0, xlens in zip(psi0s, [-dx, dx])]
        Psi = lenses.CombinedLens({'lenses':sublenses})
        
        
        fig,ax= plt.subplots(1, 1)
        ax = plot_singular_curves(ax, Psi, CausticInterpolation = True)
        
        # when e.g. dx = 0.45, better provide a fine mesh to avoid spurious results, e.g. :
        # ax = plot_singular_curves(ax, Psi, mesh = np.meshgrid(np.linspace(-1.7, 1, 500),np.linspace(-1.2, 1.2, 500)), CausticInterpolation = True)

      ```
    """
    

    if (mesh[0] is not None)&(mesh[1] is not None):
        X1 = mesh[0]
        X2 = mesh[1]
        automatic_plot_range = False
        
    else:
        # Coarse grid sampling
        X1coarse, X2coarse = np.meshgrid(np.linspace(-target_range,target_range,250),     np.linspace(-target_range,target_range,250))
        detcoarse = psi.shear(X1coarse, X2coarse)['detA']
        mask = np.abs(detcoarse) < 0.1 # region near det =0
        
        # Define refined meshgrid in this region
        x1_refined = np.linspace(X1coarse[mask].min()-0.2*abs(X1coarse[mask].min()), X1coarse[mask].max()+0.2*abs(X1coarse[mask].max()), 300)
        x2_refined = np.linspace(X2coarse[mask].min()-0.2*abs(X2coarse[mask].min()), X2coarse[mask].max()+0.2*abs(X2coarse[mask].max()), 300) 
        X1, X2 = np.meshgrid(x1_refined, x2_refined)
        
        automatic_plot_range = True
       
    
    det = psi.shear(X1, X2)['detA']
    
    # Find zero-contour lines
    contour = ax.contour(X1, X2, det, levels=[0.0],algorithm='mpl2014', colors = 'k') #algorithm{'mpl2005', 'mpl2014', 'serial', 'threaded'}

    hasbeenlegended = False
    xmin_tmp = contour.allsegs[0][0][0][0]
    ymin_tmp = contour.allsegs[0][0][0][1]
    xmax_tmp = contour.allsegs[0][0][0][0]
    ymax_tmp = contour.allsegs[0][0][0][1]  
    
    for crit_curve in contour.allsegs[0]:  # Iterate over critical curves
        xcrit = crit_curve[:, 0]
        ycrit = crit_curve[:, 1]
            
        if CausticInterpolation:
            # s is a cumulative sum of distances, arc length -like parameter along the curve
            s = np.concatenate(([0], np.cumsum( np.sqrt(np.diff(xcrit)**2 + np.diff(ycrit)**2))))  
            
            if method == 'spline':
                # Interpolate x and y separately using periodic cubic splines
                interp_x = CubicSpline(s, xcrit, bc_type='periodic')
                interp_y = CubicSpline(s, ycrit, bc_type='periodic')
            else:
                # Interpolate x and y separately using linear interpolation
                interp_x = interp1d(s, xcrit)
                interp_y = interp1d(s, ycrit)
            
            # Generate interpolated points, the initial nb of points set internally by contour is increased by a factor pointsIncrease
            s_fine = np.linspace(0, s[-1], len(xcrit)*pointsIncrease)  
            xcrit = interp_x(s_fine)
            ycrit = interp_y(s_fine)
            
        caustic_x, caustic_y = lens_eq(psi,xcrit,ycrit)
    
        if not hasbeenlegended:
            ax.plot(xcrit[1:3], ycrit[1:3],c= 'k', label='critical curve')
            ax.plot(caustic_x, caustic_y,c= 'b', label='caustic')
            hasbeenlegended = True  
            
        else:
            ax.plot(caustic_x, caustic_y,c= 'b')

        if automatic_plot_range:
            
            if min(caustic_x.min(), xcrit.min()) < xmin_tmp:
                xmin_tmp = min(caustic_x.min(), xcrit.min())
            if min(caustic_y.min(), ycrit.min()) < ymin_tmp:
                ymin_tmp = min(caustic_y.min(), ycrit.min())
            if max(caustic_x.max(), xcrit.max()) > xmax_tmp:
                xmax_tmp = max(caustic_x.max(), xcrit.max())
            if max(caustic_y.max(), ycrit.max()) > ymax_tmp:
                ymax_tmp = max(caustic_y.max(), ycrit.max())
                
    if automatic_plot_range:        
        ax.set_xlim(xmin_tmp-0.1*(xmax_tmp-xmin_tmp), xmax_tmp+0.1*(xmax_tmp-xmin_tmp))
        ax.set_ylim(ymin_tmp-0.1*(ymax_tmp-ymin_tmp), ymax_tmp+0.1*(ymax_tmp-ymin_tmp))
    ax.legend()
    return ax

def get_singular_curves(psi, mesh = (None, None), target_range = 10, CausticInterpolation = True, pointsIncrease = 5, method = 'spline'):
    """
    returns point coordinates defining the caustics and critical curves of the glow lens defined by `psi`.

    Parameters
    ----------
    psi : glow lens
        The lens potential function.
    mesh : tuple of np.ndarray, optional
        A 2D NumPy meshgrid used to compute the critical curves. If not provided, 
        the function will automatically generate a mesh within the range (-target_range, target_range) in both x and y.
    target_range : float, optional
        The (half) range within which the function will search for critical curves if `mesh` is not provided (default is 10).
    CausticInterpolation : bool, optional
        If True, interpolation is applied to the critical curves before computing the caustics, improving smoothness (default is True).
    pointsIncrease : int, optional
        The factor by which the initial number of sampled points is increased when interpolation is enabled (default is 5).
    method : {'spline', 'linear'}, optional
        The interpolation method applied when `CausticInterpolation` is True. If 'spline', splines are used; 
        otherwise, linear interpolation is applied (default is 'spline').
         
    Returns
    -------
    array_xcrit, array_ycrit, array_caustic_x, array_caustic_y : arrays of arrays
    array_xcrit is an array containing as many arrays as there are singular curves.
    Each of its sub-arrays is the set of x-components of sampled points on 1 critical curve.
    array_ycrit, array_caustic_x, array_caustic_y are analogues for y-components and caustics, respectively.

    Example
    -----
      The following example demonstrates a specific case of lens creation and use of the function:
  
      ```python
      
        from glow import lenses
        
        xs = [[0.3, 0], [-0.6, 0.3], [0.3, -0.3], [0, 0]]
        psi0 = 1./len(xs)
        rc = 0.05
        Psis = [lenses.Psi_offcenterCIS({'psi0':psi0, 'rc':rc, 'xc1':x[0], 'xc2':x[1]}) for x in xs]
        Psi = lenses.CombinedLens({'lenses':Psis})
        
        arr_crit_x, arr_crit_y, arr_caust_x, arr_caust_y = get_singular_curves(Psi, CausticInterpolation = False)
        crit_curves = plt.plot(*[component for curve in zip(arr_crit_x, arr_crit_y)for component in curve], c='k') 
        caustics = plt.plot(*[component for curve in zip(arr_caust_x, arr_caust_y)for component in curve], c='b')
        
        plt.legend([crit_curves[0], caustics[0]], ['critical curve', 'caustic'])  

      ```
    """
   
    if (mesh[0] is not None)&(mesh[1] is not None):
        X1 = mesh[0]
        X2 = mesh[1]
        
    else:
        # Coarse grid sampling
        X1coarse, X2coarse = np.meshgrid(np.linspace(-target_range,target_range,200),     np.linspace(-target_range,target_range,200))
        detcoarse = psi.shear(X1coarse, X2coarse)['detA']
        mask = np.abs(detcoarse) < 0.1 # region near det =0
        
        # Define refined meshgrid in this region
        x1_refined = np.linspace(X1coarse[mask].min()-0.2*abs(X1coarse[mask].min()), X1coarse[mask].max()+0.2*abs(X1coarse[mask].max()), 300)
        x2_refined = np.linspace(X2coarse[mask].min()-0.2*abs(X2coarse[mask].min()), X2coarse[mask].max()+0.2*abs(X2coarse[mask].max()), 300) 
        X1, X2 = np.meshgrid(x1_refined, x2_refined)
    
    det = psi.shear(X1, X2)['detA']
    
    fig,ax = plt.subplots(1, 1)
    
    # Find zero-contour lines
    contour = ax.contour(X1, X2, det, levels=[0.0],algorithm='mpl2014') #algorithm{'mpl2005', 'mpl2014', 'serial', 'threaded'}
    plt.close(fig)
    
    array_xcrit = []
    array_ycrit = []
    array_caustic_x = []
    array_caustic_y = []

    for crit_curve in contour.allsegs[0]:  # Iterate over critical curves
        xcrit = crit_curve[:, 0]
        ycrit = crit_curve[:, 1]
            
        if CausticInterpolation:
            # s is a cumulative sum of distances, arc length -like parameter along the curve
            s = np.concatenate(([0], np.cumsum( np.sqrt(np.diff(xcrit)**2 + np.diff(ycrit)**2)))) 
            
            if method == 'spline':
                # Interpolate x and y separately using periodic cubic splines
                interp_x = CubicSpline(s, xcrit, bc_type='periodic')
                interp_y = CubicSpline(s, ycrit, bc_type='periodic')
            else:
                # Interpolate x and y separately using linear interpolation
                interp_x = interp1d(s, xcrit)
                interp_y = interp1d(s, ycrit)
            
            # Generate interpolated points, the initial nb of points set internally by contour is increased by a factor pointsIncrease
            s_fine = np.linspace(0, s[-1], len(xcrit)*pointsIncrease)  
            xcrit = interp_x(s_fine)
            ycrit = interp_y(s_fine)
            
        caustic_x, caustic_y = lens_eq(psi,xcrit,ycrit)
        array_xcrit.append(xcrit)
        array_ycrit.append(ycrit)
        array_caustic_x.append(caustic_x)
        array_caustic_y.append(caustic_y)
            
    return array_xcrit, array_ycrit, array_caustic_x, array_caustic_y
