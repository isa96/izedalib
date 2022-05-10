import matplotlib
import matplotlib.pyplot as plt


class Tampilan3D:
    @staticmethod
    def Reset():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    @staticmethod
    def Kertas(loc='best', classic=True, figsize=[13.3, 8]):
        if classic:
            plt.style.use('classic')
        params = {
            "axes.formatter.useoffset": False,
            "font.family": "sans-serif",
            "font.sans-serif": "Arial",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            'axes.titlesize': 16,
            'axes.titleweight': "bold",
            "figure.dpi": 300,
            "figure.figsize": figsize,
            "legend.loc": loc,
            "legend.fontsize": 16,
            "legend.fancybox": True,
            "mathtext.fontset": 'custom',
            "mathtext.default": 'regular',
            "figure.autolayout": True,
            "patch.edgecolor": "#000000",
            "text.color": "#000000",
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
        matplotlib.rcParams.update(params)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('xkcd:white')
        return fig, ax