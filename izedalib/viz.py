import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from izedalib.plot2D import Tampilan
from izedalib.plot3D import Tampilan3D
from izedalib.extra import ConfidenceEllipse, Warna, Tanda


def boxplot(data, id_vars, value_vars, hue=None, hue_order=None, **options):
    xlabel = options.get('xlabel', None)
    ylabel = options.get('ylabel', None)
    showfliers = options.get('showfliers', False)
    palette = options.get('palette', "pastel")
    loc = options.get('loc', 'best')
    rot = options.get('rot', 0)
    legend = options.get('legend', True)

    data_melt = pd.melt(data, id_vars=id_vars, value_vars=value_vars)

    fig, ax = Tampilan().Kertas()
    ax = sns.boxplot(data=data_melt, x='variable', y='value', hue=hue,
                     hue_order=hue_order, showfliers=showfliers, palette=palette)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=loc)
    ax.legend_.set_title(None)
    if not legend:
        ax.legend_.remove()
    plt.xticks(rotation=rot)
    return fig, ax


class MyPCA:
    
    def __init__(self, **options):
        self.x = None
        self.y = None
        self.vardf = pd.DataFrame()
        self.pcadf = pd.DataFrame()
        self.eigpc = pd.DataFrame()
        self.round_ = options.get('round_', 1)
        self.featurename = options.get('featurename', None)
        self.scaler = options.get('scaler', StandardScaler())
        self.colors = options.get('colors', Warna())
        self.markers = options.get('markers', Tanda())
        self.pca = PCA()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.pca!r})'
        )

    def fit(self, x, y):
        self.x = x
        self.y = y

        if self.scaler is not None:
            scaler = self.scaler
            self.x = scaler.fit_transform(self.x)

        self.pca = PCA()
        self.pca.fit(self.x)
        pcscore = self.pca.transform(self.x)
        pcname = [f'PC{i + 1}' for i in range(pcscore.shape[1])]
        if self.featurename is None:
            self.featurename = [
                f'Feature{i + 1}' for i in range(self.x.shape[1])]
        # var_exp = [round(i * 100, self.round_) for i in sorted(self.pca.explained_variance_ratio_, reverse=True)]
        var_exp = np.round(
            self.pca.explained_variance_ratio_ * 100, decimals=self.round_)
        self.vardf = pd.DataFrame({'Var (%)': var_exp, 'PC': pcname})
        # pcscore = self.pca.transform(self.x)
        pcaDF = pd.DataFrame(data=pcscore, columns=pcname)
        Y = pd.DataFrame(data=self.y, columns=['label'])
        self.pcadf = pd.concat([pcaDF, Y], axis=1)
        self.eigpc = pd.DataFrame(data=np.transpose(
            self.pca.components_), columns=pcname, index=self.featurename)
        return self.pca

    def GetVar(self):
        return self.pcadf, self.vardf, self.eigpc

    def GetComponents(self):
        loading_score = pd.DataFrame(
            data=self.pca.components_, columns=[self.featurename])
        return loading_score

    def GetBestFeature(self, PC=0, n=3):
        loading_score = pd.Series(
            self.pca.components_[PC], index=self.featurename)
        sorted_loading_score = loading_score.abs().sort_values(ascending=False)
        top_score = sorted_loading_score[0:n].index.values
        print(loading_score[top_score])

    def Plot(self, **options):
        PC = options.get('PC', ['PC1', 'PC2'])
        s = options.get('size', 90)
        elip = options.get('ellipse', True)
        ascending = options.get('ascending', True)
        legend = options.get('legend', True)
        loc = options.get('loc', 'best')

        self.pcadf = self.pcadf.sort_values(by=['label'], ascending=ascending)

        targets = list(self.pcadf['label'].unique())

        if len(targets) > 10:
            raise ValueError(str(targets))

        colors = self.colors[:len(targets)]
        markers = self.markers[:len(targets)]

        xlabs = f'{PC[0]} ({float(self.vardf.values[self.vardf["PC"] == PC[0], 0])}%)'
        ylabs = f'{PC[1]} ({float(self.vardf.values[self.vardf["PC"] == PC[1], 0])}%)'

        fig, ax = Tampilan().Kertas()
        for target, color, mark in zip(targets, colors, markers):
            indicesToKeep = self.pcadf['label'] == target
            x = self.pcadf.loc[indicesToKeep, PC[0]]
            y = self.pcadf.loc[indicesToKeep, PC[1]]
            ax.scatter(x, y, c=color, marker=mark, s=s, label=str(target))

            if elip:
                ConfidenceEllipse(x, y, ax, edgecolor=color)

        ax.set_xlabel(xlabs)
        ax.set_ylabel(ylabs)
        if legend:
            ax.legend(loc=loc)
        return fig

    def screenplot(self, **options):
        lim = options.get('PC', None)

        if lim is None:
            data_ = self.vardf
        else:
            data_ = self.vardf.loc[:lim, :]

        fig, _ = Tampilan().Kertas()
        plt.bar(x='PC', height='Var (%)', data=data_)
        plt.xticks(rotation='vertical')
        plt.xlabel('Principal Component')
        plt.ylabel('Percentage of Variance')
        return fig


class MyLDA:

    def __init__(self, **options):
        self.x = None
        self.xval = None
        self.y = None
        self.yval = None
        self.ldaval = None
        self.dual = None

        self.round_ = options.get("round_", 1)
        self.vardf = pd.DataFrame()
        self.ldadf = pd.DataFrame()
        self.lda = LinearDiscriminantAnalysis()
        self.scaler = options.get("scaler", StandardScaler())
        self.colors = options.get("colors", Warna())
        self.markers = options.get("markers", Tanda())
        self.cv = options.get("cv", 10)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self.lda!r})"

    def fit(self, *arrays):
        if len(arrays) == 2:
            self.x = arrays[0]
            self.y = arrays[1]
            self.dual = False
        else:
            self.x = arrays[0]
            self.xval = arrays[1]
            self.y = arrays[2]
            self.yval = arrays[3]
            self.ldaval = None
            self.dual = True

        scaler = self.scaler
        X = scaler.fit_transform(self.x)
        self.lda.fit(X, self.y)
        ldax = self.lda.transform(X)

        ldname = [f"LD{i + 1}" for i in range(ldax.shape[1])]
        self.ldadf = pd.DataFrame(ldax, columns=ldname)
        Y = pd.DataFrame(data=self.y, columns=["label"])
        self.ldadf = pd.concat([self.ldadf, Y], axis=1)

        tot = sum(self.lda.explained_variance_ratio_)
        var_exp = [
            round((i / tot) * 100, self.round_)
            for i in sorted(self.lda.explained_variance_ratio_, reverse=True)
        ]
        self.vardf = pd.DataFrame({"Var (%)": var_exp, "LD": ldname})

        if self.dual:
            Xval = scaler.transform(self.xval)
            ldax = self.lda.transform(Xval)
            self.ldaval = pd.DataFrame(ldax, columns=ldname)
            Y = pd.DataFrame(data=self.yval, columns=["label"])
            self.ldaval = pd.concat([self.ldaval, Y], axis=1)

    def getvarld(self):
        if self.dual:
            ldaDF1 = pd.concat(
                [
                    self.ldadf,
                    pd.DataFrame(
                        data=self.ldadf["label"].values, columns=["Class"]),
                ],
                axis=1,
            )
            ldaDF1["Class"] = "Training"

            ldaDF2 = pd.concat(
                [
                    self.ldaval,
                    pd.DataFrame(
                        data=self.ldaval["label"].values, columns=["Class"]),
                ],
                axis=1,
            )
            ldaDF2["Class"] = "Testing"

            ldaDF = pd.concat([ldaDF1, ldaDF2], axis=0)
        else:
            ldaDF = self.ldadf

        return ldaDF, self.vardf

    def getscore(self):
        from sklearn.model_selection import cross_val_score

        return cross_val_score(LinearDiscriminantAnalysis(), self.x, self.y, cv=self.cv)

    def plotlda(self, **options):
        elip = options.get("ellipse", True)
        ascending = options.get("ascending", True)
        legend = options.get("legend", True)
        loc = options.get("loc", "best")

        self.ldadf = self.ldadf.sort_values(by=["label"], ascending=ascending)
        nlabel = np.unique(self.y)
        if len(nlabel) < 3:
            fig, ax = Tampilan().Kertas()
            s = options.get("size", 10)

            if self.dual:
                self.ldaval = self.ldaval.sort_values(
                    by=["label"], ascending=ascending)
                ax = sns.stripplot(
                    x="label", y="LD1", color="k", size=s, data=self.ldadf
                )
                ax = sns.stripplot(
                    x="label",
                    y="LD1",
                    marker="^",
                    color="red",
                    size=s,
                    data=self.ldaval,
                )
            else:
                ax = sns.stripplot(x="label", y="LD1", size=s, data=self.ldadf)

            ax.set_xlabel("Classes")
            ax = plt.axhline(y=0, linewidth=1.5, color="black", linestyle="--")
            return fig
        else:
            targets = list(self.ldadf["label"].unique())

            s = options.get("size", 90)
            if len(targets) > 10:
                raise ValueError(str(targets))

            colors = self.colors[: len(targets)]
            markers = self.markers[: len(targets)]

            xlabs = f"LD1 ({self.vardf.values[0, 0]}%)"
            ylabs = f"LD2 ({self.vardf.values[1, 0]}%)"

            fig, ax = Tampilan().Kertas()
            for target, color, mark in zip(targets, colors, markers):
                indicesToKeep = self.ldadf["label"] == target
                x = self.ldadf.loc[indicesToKeep, "LD1"]
                y = self.ldadf.loc[indicesToKeep, "LD2"]
                ax.scatter(x, y, c=color, marker=mark, s=s, label=target)

                if elip:
                    ConfidenceEllipse(x, y, ax, edgecolor=color)

            if self.dual:
                self.ldaval = self.ldaval.sort_values(
                    by=["label"], ascending=ascending)

                for target, color, mark in zip(targets, colors, markers):
                    indicesToKeep = self.ldaval["label"] == target
                    x = self.ldaval.loc[indicesToKeep, "LD1"]
                    y = self.ldaval.loc[indicesToKeep, "LD2"]
                    ax.scatter(
                        x,
                        y,
                        marker=mark,
                        s=s,
                        facecolors="none",
                        edgecolors=color,
                        label=f"{target} - test",
                    )

            if legend:
                ax.legend(loc=loc)
            ax.set_xlabel(xlabs)
            ax.set_ylabel(ylabs)

            return fig

class MyPCA3D:
    
    def __init__(self, **options):
        self.x = None
        self.y = None
        self.vardf = pd.DataFrame()
        self.pcadf = pd.DataFrame()
        self.eigpc = pd.DataFrame()
        self.round_ = options.get('round_', 1)
        self.featurename = options.get('featurename', None)
        self.scaler = options.get('scaler', StandardScaler())
        self.colors = options.get('colors', Warna())
        self.markers = options.get('markers', Tanda())
        self.pca = PCA()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.pca!r})'
        )

    def fit(self, x, y):
        self.x = x
        self.y = y

        if self.scaler is not None:
            scaler = self.scaler
            self.x = scaler.fit_transform(self.x)

        self.pca = PCA()
        self.pca.fit(self.x)
        pcscore = self.pca.transform(self.x)
        pcname = [f'PC{i + 1}' for i in range(pcscore.shape[1])]
        if self.featurename is None:
            self.featurename = [
                f'Feature{i + 1}' for i in range(self.x.shape[1])]
        # var_exp = [round(i * 100, self.round_) for i in sorted(self.pca.explained_variance_ratio_, reverse=True)]
        var_exp = np.round(
            self.pca.explained_variance_ratio_ * 100, decimals=self.round_)
        self.vardf = pd.DataFrame({'Var (%)': var_exp, 'PC': pcname})
        # pcscore = self.pca.transform(self.x)
        pcaDF = pd.DataFrame(data=pcscore, columns=pcname)
        Y = pd.DataFrame(data=self.y, columns=['label'])
        self.pcadf = pd.concat([pcaDF, Y], axis=1)
        self.eigpc = pd.DataFrame(data=np.transpose(
            self.pca.components_), columns=pcname, index=self.featurename)
        return self.pca

    def GetVar(self):
        return self.pcadf, self.vardf, self.eigpc

    def GetComponents(self):
        loading_score = pd.DataFrame(
            data=self.pca.components_, columns=[self.featurename])
        return loading_score

    def GetBestFeature(self, PC=0, n=3):
        loading_score = pd.Series(
            self.pca.components_[PC], index=self.featurename)
        sorted_loading_score = loading_score.abs().sort_values(ascending=False)
        top_score = sorted_loading_score[0:n].index.values
        print(loading_score[top_score])

    def Plot(self, **options):
        PC = options.get('PC', ['PC1', 'PC2', 'PC3'])
        s = options.get('size', 90)
        elip = options.get('ellipse', True)
        ascending = options.get('ascending', True)
        legend = options.get('legend', True)
        loc = options.get('loc', 'best')

        self.pcadf = self.pcadf.sort_values(by=['label'], ascending=ascending)

        targets = list(self.pcadf['label'].unique())

        if len(targets) > 10:
            raise ValueError(str(targets))

        colors = self.colors[:len(targets)]
        markers = self.markers[:len(targets)]

        xlabs = f'{PC[0]} ({float(self.vardf.values[self.vardf["PC"] == PC[0], 0])}%)'
        ylabs = f'{PC[1]} ({float(self.vardf.values[self.vardf["PC"] == PC[1], 0])}%)'
        zlabs = f'{PC[2]} ({float(self.vardf.values[self.vardf["PC"] == PC[2], 0])}%)'

        fig, ax = Tampilan3D().Kertas()
        for target, color, mark in zip(targets, colors, markers):
            indicesToKeep = self.pcadf['label'] == target
            x = self.pcadf.loc[indicesToKeep, PC[0]]
            y = self.pcadf.loc[indicesToKeep, PC[1]]
            z = self.pcadf.loc[indicesToKeep, PC[2]]
            ax.scatter(x, y, z, c=color, marker=mark, s=s, label=str(target))

#             if elip:
#                 ConfidenceEllipse(x, y, ax, edgecolor=color)

        ax.set_xlabel(xlabs)
        ax.set_ylabel(ylabs)
        ax.set_zlabel(zlabs)
        if legend:
            ax.legend(loc=loc)
        return fig

    def screenplot(self, **options):
        lim = options.get('PC', None)

        if lim is None:
            data_ = self.vardf
        else:
            data_ = self.vardf.loc[:lim, :]

        fig, _ = Tampilan3D().Kertas()
        plt.bar(x='PC', height='Var (%)', data=data_)
        plt.xticks(rotation='vertical')
        plt.xlabel('Principal Component')
        plt.ylabel('Percentage of Variance')
        plt.zlabel('Percentage of Variance')
        return fig