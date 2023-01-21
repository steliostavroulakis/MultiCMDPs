from ipywidgets import interact, FloatSlider, IntSlider, fixed
from ipywidgets import widgets
import matplotlib.pyplot as plt

def show_plot(knob):
    plt.imshow(plt.imread('experiment_result_{}.png'.format(int(knob))))
    plt.axis('off')
    plt.savefig('experiment_result.png')
    plt.close()

knob = FloatSlider(min=10, max=100, step=90, continuous_update=False)
widgets.interact(show_plot, knob=knob)


