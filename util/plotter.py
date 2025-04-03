import matplotlib.pyplot as plt
import seaborn as sns
import torch

FONT_SIZE = 20
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams['lines.linewidth'] = 2

class Plotter(object):
    def __init__(self):
        # sns.set_theme(style='darkgrid')
        pass

    def plot_hist(self, data, saved_path, x_line=None):
        flatten = data.flatten()
        histogram = torch.histc(flatten, bins=2500)
        x = torch.linspace(flatten.min(), flatten.max(), steps=histogram.shape[0])
        plt.figure()
        plt.title("Hist")
        plt.xlabel("intensity")
        plt.ylabel("frequency")
        if x_line:
            plt.axvline(x=x_line, color='red', linestyle='--')
            plt.text(x_line, 0, f'x={x_line:5f}', color='red', ha='left', va='bottom', fontsize=10)
        plt.plot(x, histogram)
        plt.savefig(saved_path, dpi=300)
        plt.close()
    
    def plot_series(self, data, saved_path, title="Series"):
        flatten = data.flatten()
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.plot(flatten)

        # todo: formulate to callback func...
        plt.xlabel("Iterations")
        plt.ylabel("Intersection Ratio")
        plt.ylim(0,1) 

        plt.savefig(saved_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_mutil_series(self, data_list, saved_path):
        plt.figure(figsize=(10, 5))
        plt.title("Series")
        for data in data_list:
            flatten = data.flatten()
            plt.plot(flatten)
        plt.savefig(saved_path, dpi=300)
        plt.close()
    
    def plot_heat_map(self, data, saved_path, camp="crest"):
        plt.figure(figsize=(10, 10))
        plt.title("Error Heatmap")
        sns.heatmap(data, cmap=camp)
        plt.savefig(saved_path, dpi=300)
        plt.close()





# export singleton 
plotter = Plotter()