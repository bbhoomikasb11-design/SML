"""Publication figures from the saved experiment; no plotted synthetic observations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import probplot
from statsmodels.nonparametric.smoothers_lowess import lowess

from config import FEATURES, ROOT
from src.feature_engineering import with_analysis_features

SHORT = ["Cement", "Slag", "Fly ash", "Water", "Plasticizer", "Coarse agg.", "Fine agg.", "Age"]


def export_figures(bundle, directory=None):
    directory = directory or ROOT/"artifacts/figures"/bundle["metadata"]["experiment_id"]
    directory.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid",font_scale=.9,rc={"axes.spines.top":False,"axes.spines.right":False})
    data = with_analysis_features(bundle["data"])
    files = {}
    def save(name,fig):
        fig.tight_layout(pad=1.4)
        path = directory/f"{name}.png"
        fig.savefig(path,dpi=180,bbox_inches="tight",facecolor="white")
        plt.close(fig)
        files[name]=path
    fig,ax = plt.subplots(figsize=(9,3.8))
    sns.histplot(data.Strength,bins=30,kde=True,ax=ax,color="#2563EB")
    ax.set(xlabel="Compressive strength (MPa)",ylabel="Observations",title="Observed concrete compressive strength")
    save("strength_distribution",fig)
    fig,axes = plt.subplots(4,2,figsize=(9,10))
    for f,ax in zip(FEATURES,axes.flat):
        sns.histplot(data[f],bins=24,ax=ax,color="#2563EB")
        ax.set(xlabel=f+ (" (days)" if f=="Age" else " (kg/m³)"),ylabel="Observations",title=f)
    save("input_histograms",fig)
    fig,ax = plt.subplots(figsize=(9,6))
    sns.heatmap(data[FEATURES+["Strength"]].corr(),xticklabels=SHORT+["Strength"],yticklabels=SHORT+["Strength"],
                annot=True,fmt=".2f",cmap="RdBu_r",vmin=-1,vmax=1,ax=ax,cbar_kws={"label":"Pearson r"})
    ax.set_title("Pearson correlations among original variables")
    save("correlation",fig)
    fig,axes = plt.subplots(2,4,figsize=(12,6))
    for f,ax in zip(FEATURES,axes.flat):
        ax.scatter(data[f],data.Strength,s=6,alpha=.4,color="#2563EB")
        ax.set(xlabel=f+ (" (days)" if f=="Age" else " (kg/m³)"),ylabel="Strength (MPa)",title=f)
    save("feature_scatter",fig)
    fig,axes = plt.subplots(2,4,figsize=(12,5.5))
    for f,ax in zip(FEATURES,axes.flat):
        sns.boxplot(y=data[f],ax=ax,color="#93C5FD",width=.4)
        ax.set(ylabel="days" if f=="Age" else "kg/m³",title=f)
    save("boxplots",fig)
    selected = data[["Cement","Water","Age","Strength"]].rename(columns={"Cement":"Cement (kg/m³)","Water":"Water (kg/m³)","Age":"Age (days)","Strength":"Strength (MPa)"})
    grid = sns.pairplot(selected,corner=True,plot_kws={"s":8,"alpha":.3,"color":"#2563EB"},height=1.9)
    save("pairwise",grid.fig)
    metrics = bundle["metrics"]
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    x = np.arange(len(metrics))
    for i,key in enumerate(["Train RMSE","Test RMSE","CV RMSE"]):
        axes[0].bar(x+(i-1)*.23,metrics[key],.23,label=key,color=["#93C5FD","#2563EB","#14B8A6"][i])
    axes[0].set_xticks(x,metrics.Model,rotation=15)
    axes[0].set(ylabel="RMSE (MPa)",title="Prediction error comparison")
    axes[0].legend(fontsize=8)
    sns.stripplot(data=bundle["cv_scores"],x="Model",y="RMSE",ax=axes[1],color="#2563EB",size=7,jitter=False)
    axes[1].set(ylabel="Validation RMSE (MPa)",title="Five-fold score distribution" if bundle['metadata']['folds']==5 else "CV score distribution")
    axes[1].tick_params(axis="x",rotation=15)
    save("comparison",fig)
    pred = bundle["predictions"][bundle["metadata"]["selected"]]
    fig,ax = plt.subplots(figsize=(7,4.5))
    ax.scatter(pred.Actual,pred.Predicted,color="#2563EB",s=17,alpha=.65,label="Held-out observations")
    bounds = [min(pred.Actual.min(),pred.Predicted.min()),max(pred.Actual.max(),pred.Predicted.max())]
    ax.plot(bounds,bounds,"--",color="#14B8A6",label="Perfect prediction")
    ax.set(xlabel="Actual strength (MPa)",ylabel="Predicted strength (MPa)",title="Selected model on held-out observations")
    ax.legend()
    save("actual_predicted",fig)
    fig,axes = plt.subplots(2,2,figsize=(10,7))
    for ax,xcol in [(axes[0,0],"Predicted"),(axes[1,1],"Actual")]:
        ax.scatter(pred[xcol],pred.Residual,s=13,alpha=.55,color="#2563EB")
        trend = lowess(pred.Residual,pred[xcol],frac=.5)
        ax.plot(trend[:,0],trend[:,1],color="#14B8A6",label="LOWESS")
        ax.axhline(0,color="#64748B",ls="--")
        ax.set(xlabel=f"{xcol} strength (MPa)",ylabel="Residual (MPa)",title=f"Residuals versus {xcol.lower()}")
        ax.legend()
    sns.histplot(pred.Residual,bins=25,ax=axes[0,1],color="#2563EB")
    axes[0,1].set(xlabel="Residual (MPa)",ylabel="Observations",title="Residual distribution")
    probplot(pred.Residual,dist="norm",plot=axes[1,0])
    axes[1,0].set(xlabel="Theoretical normal quantile",ylabel="Ordered residual (MPa)",title="Normal Q–Q plot")
    save("residuals",fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax,f,title in [(axes[0],"Water_Cement_Ratio","Water–cement ratio (kg/kg)"),(axes[1],"Age","Age (days)")]:
        ax.scatter(data[f],data.Strength,s=9,alpha=.3,color="#2563EB",label="Observed")
        trend = lowess(data.Strength,data[f],frac=.6 if f == "Age" else .35)
        ax.plot(trend[:,0],trend[:,1],color="#0D9488",lw=2,label="LOWESS")
        ax.set(xlabel=title,ylabel="Strength (MPa)",title=f"Observed {title.lower()}")
        ax.legend()
    save("factor_trends",fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    sns.histplot(data.Water_Cement_Ratio,bins=30,ax=axes[0],color="#2563EB")
    axes[0].set(xlabel="Water–cement ratio (kg/kg)",ylabel="Observations",title="Water–cement ratio distribution")
    sns.boxplot(data=data,x="Age Group",y="Strength",ax=axes[1],color="#99F6E4")
    axes[1].set(xlabel="Curing age group (days)",ylabel="Strength (MPa)",title="Strength by observed age group")
    save("factor_distributions",fig)
    return files


if __name__=="__main__":
    from src.model_training import load_experiment
    print(export_figures(load_experiment()))
