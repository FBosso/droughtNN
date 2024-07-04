"""
======================================
Radar chart (aka spider or star chart)
======================================

This example creates a radar chart, also known as a spider or star chart [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


################## ELM VS lin ##################

def example_data():

    data = [
        ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        ('Monthly ML VS Monthly Linear', [
            [345.35,189.20,344.41,261.36,293.82,242.66,539.56,333.38,340.40,271.95,214.30,234.29],
            [583.49,220.80,546.88,362.45,450.56,412.92,730.55,453.24,393.58,305.27,266.91,359.46],
            [1103.933979899311, 930.932921164406, 550.9031041995327, 958.2714413246738, 752.599650455287, 812.0042593192302, 2201.0436057765082, 3702.955978750428, 1626.1348474040424, 2310.859252410873, 1313.640876390141, 1922.1947722996397]]),
        ]
    return data


if __name__ == '__main__':
    
    
  

    N = 9
    theta = radar_factory(12, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axes = plt.subplots(figsize=(9, 5), nrows=1, ncols=2,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'white']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axes.flat, data):
        ax.set_rgrids([0,600,1200,1800,2400,3200,3800]) 
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            
            if color == 'white':
                ax.plot(theta, d, color=color, visible=False)
                ax.fill(theta, d, facecolor=color, alpha=0)
            else:
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=0.25)
                
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    ax = axes[0]
    labels = ('Monthly ML', 'Monthly Linear')
    #legend = ax.legend(labels, loc=(1, 1),labelspacing=0.1, fontsize='small', edgecolor="white")

    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    
    plt.savefig('elm _VS_lin.pdf')

    plt.show()
    
    
    
################## FFNN VS lin ##################

def example_data():

    data = [
        ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        
        ('Yearly ML VS Yearly Linear', [
            [555.04,308.45,512.22,327.65,409.15,495.77,978.36,353.21,719.84,547.09,189.62,450.75],
            [1379.66,951.66, 763.67, 1139.72, 840.51, 743.96, 1716.77, 1166.03, 1332.39, 1415.54, 386.66, 1145.63],
            [1103.933979899311, 930.932921164406, 550.9031041995327, 958.2714413246738, 752.599650455287, 812.0042593192302, 2201.0436057765082, 3702.955978750428, 1626.1348474040424, 2310.859252410873, 1313.640876390141, 1922.1947722996397]
            ]),
        
    
        
        ]
    return data


if __name__ == '__main__':
    
    
  

    N = 9
    theta = radar_factory(12, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axes = plt.subplots(figsize=(9, 5), nrows=1, ncols=2,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['g', 'orange', 'white']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axes.flat, data):
        ax.set_rgrids([0,600,1200,1800,2400,3200,3800])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            if color == 'white':
                ax.plot(theta, d, color=color, visible=False)
                ax.fill(theta, d, facecolor=color, alpha=0)
            else:
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=0.25)
                
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    ax = axes[0]
    labels = ('Yearly ML', 'Yearly Linear')
    #legend = ax.legend(labels, loc=(1, 1),labelspacing=0.1, fontsize='small', edgecolor="white")

    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    
    plt.savefig('ffnn_VS_lin.pdf')

    plt.show()


################## ML models VS ECMWF ##################

def example_data():

    data = [
        ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        
        ('Monthly ML VS Yearly ML VS ECMWF' , [
            [345.35,189.20,344.41,261.36,293.82,242.66,539.56,333.38,340.40,271.95,214.30,234.29],
            [555.04,308.45,512.22,327.65,409.15,495.77,978.36,353.21,719.84,547.09,189.62,450.75],
            
            #[1217.5,1336.82,1407.62,922.78,940.5,1306.98,1403.31,2129.32,2183.98,2112.26,1675.97,1544.66]
            [1103.933979899311, 930.932921164406, 550.9031041995327, 958.2714413246738, 752.599650455287, 812.0042593192302, 2201.0436057765082, 3702.955978750428, 1626.1348474040424, 2310.859252410873, 1313.640876390141, 1922.1947722996397]
            ]),
        
    
        
        ]
    return data


if __name__ == '__main__':
    
    
  

    N = 9
    theta = radar_factory(12, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axes = plt.subplots(figsize=(9, 5), nrows=1, ncols=2,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b','g','purple']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axes.flat, data):
        ax.set_rgrids([0,600,1200,1800,2400,3200,3800])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    ax = axes[0]
    labels = ('Monthly ML','Yearly ML','ECMWF')
    #legend = ax.legend(labels, loc=(1, 1),labelspacing=0.1, fontsize='small', edgecolor="white")

    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    
    plt.savefig('ML_VS_ECMWF.pdf')

    plt.show()
    

    
################## Full legend ##################

def example_data():

    data = [
        ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        
        ('Total comparison', [
            [345.35,189.20,344.41,261.36,293.82,242.66,539.56,333.38,340.40,271.95,214.30,234.29],
            [308.45,512.22,327.65,409.15,495.77,978.36,353.21,719.84,547.09,189.62,450.75,555.04],
            
            [583.49,220.80,546.88,362.45,450.56,412.92,730.55,453.24,393.58,305.27,266.91,359.46],
            [951.66, 763.67, 1139.72, 840.51, 743.96, 1716.77, 1166.03, 1332.39, 1415.54, 386.66, 1145.63, 1379.66],
            [1103.933979899311, 930.932921164406, 550.9031041995327, 958.2714413246738, 752.599650455287, 812.0042593192302, 2201.0436057765082, 3702.955978750428, 1626.1348474040424, 2310.859252410873, 1313.640876390141, 1922.1947722996397]
            ]),
        
    
        
        ]
    return data


if __name__ == '__main__':
    
    
  

    N = 9
    theta = radar_factory(12, frame='polygon')

    data = example_data()
    spoke_labels = data.pop(0)

    fig, axes = plt.subplots(figsize=(9, 5), nrows=1, ncols=2,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b','g','r','orange','purple']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axes.flat, data):
        ax.set_rgrids([0,300,600,900,1200,1500,1800])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    ax = axes[0]
    labels = ('Monthly ML','Yearly ML','Monthly Linear','Yearly Linear', 'ECMWF')
    legend = ax.legend(labels, loc=(1, 1),
                       labelspacing=0.1, fontsize='small', edgecolor="white")

    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    
    plt.savefig('legend.pdf')

    plt.show()

