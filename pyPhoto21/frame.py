import numpy as np
import matplotlib.figure as figure
from matplotlib.widgets import Slider


class FrameViewer:
    def __init__(self, data):
        self.data = data
        self.fig = figure.Figure()
        self.ax = self.fig.add_subplot(111)

        axmax = self.fig.add_axes([0.25, 0.01, 0.65, 0.03])
        self.smax = Slider(axmax, 'Max', 0, np.max(self.data.get_num_pts()), valinit=50)
        self.ind = self.data.get_num_pts() // 2
        self.current_frame = self.data.get_display_frame()

        self.im = self.ax.imshow(self.current_frame,
                                aspect = 'auto',
                                cmap='jet')
        self.update()

    def get_slider_max(self):
        return self.smax

    def get_fig(self):
        return self.fig

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def contrast(self, event):
        print('Changing contrast')
        print(smax.val)
        self.im.set_clim([0,smax.val])
        self.update()


    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

    def update_new_image_size(self, rli=False):
        print('updating frame size...')
        self.fig.clf()
        self.im = self.ax.imshow(self.current_frame,
                                 aspect='auto',
                                 cmap='jet')
        self.ax = self.fig.add_subplot(111)

        axmax = self.fig.add_axes([0.25, 0.01, 0.65, 0.03])
        self.smax = Slider(axmax, 'Max', 0, np.max(self.data.get_num_pts()), valinit=50)
        self.ind = self.data.get_num_pts() // 2
        self.current_frame = self.data.get_display_frame(index=self.ind,
                                                         get_rli=rli)
        self.im = self.ax.imshow(self.current_frame,
                                 aspect='auto',
                                 cmap='jet')
        self.update(rli=rli)

    def update(self, rli=False):
        print('updating frame...')
        self.current_frame = self.data.get_display_frame(index=self.ind,
                                                         get_rli=rli)

        self.im.set_data(self.current_frame)
        self.im.set_clim(vmin=np.min(self.current_frame),
                         vmax=np.max(self.current_frame))

        #self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

    # def save_frame(self):
    #    plt.savefig(path + 'readout-RLI-' + image_version + ".png")

