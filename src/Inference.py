import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as  mpatches
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set(font_scale=1.5)
sns.set_theme(style='white')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr


def regression_metrics(ytrue, ypred):
    """Calculating the evaluation metric based on regression results:
    input:
            ytrue: 
            ypred:
            
    output:
            root mean square error(rmse)
            root square(r^2)
            mean absolute error(mae)
            mean absolute percent error(mape)
            mean of true yield
            mean of prediction yield
    """
    mean_ytrue = np.mean(ytrue)
    mean_ypred = np.mean(ypred)
    rmse       = np.sqrt(mean_squared_error(ytrue, ypred))
    mape       = mean_absolute_percentage_error(ytrue, ypred)
    r_square   = r2_score(ytrue, ypred)
    mae        = mean_absolute_error(ytrue, ypred)
    
    return [r_square , mae, rmse, mape, mean_ytrue, mean_ypred] 

def eval_on_three_main_label_range_pred(df, th1: int, th2: int):

    true_labels = df['ytrue'].values
    pred_labels = df['ypred_w15'].values


    #for i in range(30):

    #if i < th1: 
    true_label_C1 = true_labels[np.where((true_labels >= 0) & (true_labels <= th1))]
    pred_label_C1 = pred_labels[np.where((true_labels >= 0) & (true_labels <= th1))]

    #elif (i >= th1) & (i < th2):
    true_label_C2 = true_labels[np.where((true_labels > th1) & (true_labels < th2))]
    pred_label_C2 = pred_labels[np.where((true_labels > th1) & (true_labels < th2))]

    #elif i >= th2: 
    true_label_C3 = true_labels[np.where(true_labels >= th2)]
    pred_label_C3 = pred_labels[np.where(true_labels >= th2)]

    All_R2, All_MAE, All_RMSE, All_MAPE, _, _ = regression_metrics(true_labels, pred_labels)
    print(f"C1 num samples: {len(true_label_C1)} | C2 num samples: {len(true_label_C2)} | C3 num samples: {len(true_label_C3)} ")
    C1_R2, C1_MAE, C1_RMSE, C1_MAPE, _, _ = regression_metrics(true_label_C1, pred_label_C1)
    C2_R2, C2_MAE, C2_RMSE, C2_MAPE, _, _ = regression_metrics(true_label_C2, pred_label_C2)
    C3_R2, C3_MAE, C3_RMSE, C3_MAPE, _, _ = regression_metrics(true_label_C3, pred_label_C3)

    print(f"C1 is yield value between 0 and {th1}, C2 is yield value between {th1} and {th2}, and C3 is yield value bigger than {th2}")
    print(f"All: MAE = {All_MAE:.2f}, MAPE = {All_MAPE:.2f} | C1: MAE = {C1_MAE:.2f}, MAPE = {C1_MAPE*100:.2f} | C2: MAE = {C2_MAE:.2f}, MAPE = {C2_MAPE*100:.2f} | C3: MAE = {C3_MAE:.2f}, MAPE = {C3_MAPE*100:.2f}")
    print(f"=========================================================================================================================")
    return [C1_MAE, C1_MAPE, C2_MAE, C2_MAPE, C3_MAE, C3_MAPE]


def eval_on_extreme_main_label_range_pred(df, th1: int, th2: int):

    true_labels = df['ytrue'].values
    pred_labels = df['ypred_w15'].values


    #for i in range(30):

    #if i < th1: 
    true_label_C1 = true_labels[np.where((true_labels >= 0) & (true_labels <= th1))]
    pred_label_C1 = pred_labels[np.where((true_labels >= 0) & (true_labels <= th1))]

    #elif i >= th2: 
    true_label_C3 = true_labels[np.where(true_labels >= th2)]
    pred_label_C3 = pred_labels[np.where(true_labels >= th2)]


    #elif (i >= th1) & (i < th2):
    true_label_Cm = true_labels[np.where((true_labels > th1) & (true_labels < th2))]
    pred_label_Cm = pred_labels[np.where((true_labels > th1) & (true_labels < th2))]

    true_label_ex = []
    true_label_ex.append(true_label_C1)
    true_label_ex.append(true_label_C3)
    true_label_ex = np.concatenate(true_label_ex)
    pred_label_ex = []
    pred_label_ex.append(pred_label_C1)
    pred_label_ex.append(pred_label_C3)
    pred_label_ex = np.concatenate(pred_label_ex)


    All_R2, All_MAE, All_RMSE, All_MAPE, _, _ = regression_metrics(true_labels, pred_labels)
    print(f"Majority range yield: {len(true_label_Cm)} | Extreme yield value: {len(true_label_ex)}")

    Cm_R2, Cm_MAE, Cm_RMSE, Cm_MAPE, _, _ = regression_metrics(true_label_Cm, pred_label_Cm)
    Cex_R2, Cex_MAE, Cex_RMSE, Cex_MAPE, _, _ = regression_metrics(true_label_ex, pred_label_ex)

    #print(f"C1 is yield value between 0 and {th1}, C2 is yield value between {th1} and {th2}, and C3 is yield value bigger than {th2}")
    print(f"All: MAE = {All_MAE:.2f}, MAPE = {All_MAPE:.2f} | Cm: MAE = {Cm_MAE:.2f}, MAPE = {Cm_MAPE*100:.2f} | Cex: MAE = {Cex_MAE:.2f}, MAPE = {Cex_MAPE*100:.2f}")
    print(f"==============================================================================================================")
    return [Cm_MAE, Cm_MAPE, Cex_MAE, Cex_MAPE]


def Erroe_hist_visulization(df):

    fig, axs = plt.subplots(1, 2 , figsize = (15, 4))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    ax1 = sns.histplot(df, x = 'ytrue', bins = 30, ax = axs[0])
    ax1.set_title("Yield observation distribution")
    MAPE_Errors, counts = [], []
    for i in range(30):
        Data  = df.loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))] 
        counts.append(len(Data))
        MAPE = mean_absolute_percentage_error(Data['ytrue'], Data['ypred_w15'])
        if MAPE > 1: 
            MAPE = 1
        MAPE_Errors.append(MAPE*100)

    pearson_value = pearsonr(MAPE_Errors, counts)[0]
    bins_value  = np.arange(1, 31, 1)
    ax2 = sns.barplot(x = bins_value, y= MAPE_Errors, color = sns.color_palette()[0], width = 0.9, ax = axs[1])



    ax2.set(ylabel='MAPE Error')
    ax2.set(xlabel='yture')
    ax2.set_title(r"Pearson Correlation:{:.2f}".format(pearson_value))

    None

def Erroe_hist_visulization_V2(df):

    fig, axs = plt.subplots(1, 1 , figsize = (12, 4))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    MAPE_Errors, counts = [], []
    for i in range(1, 30):
        if i == 1:
            Data  = df.loc[(df['ytrue'] < (i+1))]
        elif i == 29: 
            Data  = df.loc[(df['ytrue'] >= (i))]
        else: 
            Data  = df.loc[(df['ytrue'] >= i) & (df['ytrue'] < (i+1))] 
        counts.append(len(Data))
        MAPE = mean_absolute_percentage_error(Data['ytrue'], Data['ypred_w15'])
        if MAPE > 1: 
            MAPE = 1
        MAPE_Errors.append(MAPE*100)

    pearson_value = pearsonr(MAPE_Errors, counts)[0]
    bins_value  = np.arange(1, 30, 1)

    ax1  = sns.barplot(x = bins_value, y= counts, color = sns.color_palette()[0], width = 0.9, ax = axs)
    axs01 = axs.twinx()
    ax01 = sns.barplot(x = bins_value, y= MAPE_Errors, color = sns.color_palette()[3], alpha = 0.5, width = 0.9, ax = axs01)


    handles = [mpatches.Patch(facecolor = sns.color_palette()[0], label = 'Number of Samples'),
    mpatches.Patch(facecolor = sns.color_palette()[3], label = 'MAPE Error')
    ]
    axs.legend(handles = handles, loc = 'upper right')


    ax1.set(ylabel='Number of Samples')
    ax01.set(ylabel='MAPE Error')
    ax1.set(xlabel='yield value')
    plt.title(r"Pearson Correlation:{:.2f}".format(pearson_value))

    None


def time_series_evaluation_plots(train, val, test, fig_save_name): 
    
    Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
    
    results = pd.DataFrame()
    train_r2_list, train_mae_list, train_rmse_list, train_mape_list = [], [], [], [] 
    val_r2_list, val_mae_list, val_rmse_list, val_mape_list         = [], [], [], []
    test_r2_list, test_mae_list, test_rmse_list, test_mape_list     = [], [], [], []
    
    for i in range(len(Weeks)): 
        train_r2, train_mae, train_rmse, train_mape, _, _ = regression_metrics(train['ytrue'], train.iloc[:, i+1])
        train_r2_list.append(train_r2)
        train_mae_list.append(train_mae)
        train_rmse_list.append(train_rmse)
        train_mape_list.append(train_mape)
        
        val_r2, val_mae, val_rmse, val_mape, _, _         = regression_metrics(val['ytrue'], val.iloc[:, i+1])
        val_r2_list.append(val_r2)
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)
        val_mape_list.append(val_mape)
        
        test_r2, test_mae, test_rmse, test_mape, _, _     = regression_metrics(test['ytrue'], test.iloc[:, i+1])
        test_r2_list.append(test_r2)
        test_mae_list.append(test_mae)
        test_rmse_list.append(test_rmse)
        test_mape_list.append(test_mape)
    
    results['weeks']      = Weeks
    results['Train_R2']   = train_r2_list
    results['Valid_R2']   = val_r2_list
    results['Test_R2']    = test_r2_list
    results['Train_MAE']  = train_mae_list
    results['Valid_MAE']  = val_mae_list
    results['Test_MAE']   = test_mae_list
    results['Train_RMSE'] = train_rmse_list
    results['Valid_RMSE'] = val_rmse_list
    results['Test_RMSE']  = test_rmse_list
    results['Train_MAPE'] = train_mape_list
    results['Valid_MAPE'] = val_mape_list
    results['Test_MAPE']  = test_mape_list
 

    #plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(2, 2, figsize = (20, 10), sharex=True)


    axs[0, 0].plot(results["weeks"], results["Train_R2"], "-o")
    axs[0, 0].plot(results["weeks"], results["Valid_R2"], "-*")
    axs[0, 0].plot(results["weeks"], results["Test_R2"],  "-d")
    axs[0, 0].set_ylabel('R2')
    axs[0, 0].set_facecolor('white')
    plt.setp(axs[0, 0].spines.values(), color='k')
    
    axs[0, 1].plot(results["weeks"], results["Train_RMSE"], "-o")
    axs[0, 1].plot(results["weeks"], results["Valid_RMSE"], "-*")
    axs[0, 1].plot(results["weeks"], results["Test_RMSE"],  "-d")
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].set_facecolor('white')
    plt.setp(axs[0, 1].spines.values(), color='k')
    
    axs[1, 0].plot(results["weeks"], results["Train_MAE"], "-o")
    axs[1, 0].plot(results["weeks"], results["Valid_MAE"], "-*")
    axs[1, 0].plot(results["weeks"], results["Test_MAE"],  "-d")
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].set_ylabel('MAE (ton/ac)')
    axs[1, 0].set_facecolor('white')
    plt.setp(axs[1, 0].spines.values(), color='k')
    
    axs[1, 1].plot(results["weeks"], results["Train_MAPE"], "-o", label = 'train')
    axs[1, 1].plot(results["weeks"], results["Valid_MAPE"], "-*", label = 'valid')
    axs[1, 1].plot(results["weeks"], results["Test_MAPE"],  "-d", label = 'test')
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].set_ylabel('MAPE (%)')
    axs[1, 1].set_facecolor('white')
    plt.setp(axs[1, 1].spines.values(), color='k')
    axs[1, 1].legend(loc="upper right")
    
    
    

    plt.savefig(fig_save_name, dpi = 300)
    
    return results 

def train_val_test_satterplot(train_df, valid_df, test_df, week = None, cmap  = None, mincnt = None, fig_save_name = None):

    
    if week == None: 
        week_pred = 'ypred'
    else: 
        week_pred = 'ypred_w' + str(week)

    w_train_e1  = train_df[['ytrue', week_pred]]
    w_train_e1  = w_train_e1.rename(columns={week_pred: "ypred"})
    tarin_true  = w_train_e1['ytrue']
    train_pred  = w_train_e1['ypred']
    tr_r2, tr_mae, tr_rmse, tr_mape, _,_ = regression_metrics(tarin_true, train_pred)

    TR = sns.jointplot(x=tarin_true, y=train_pred, kind="hex", height=8, ratio=4,  
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap , mincnt=mincnt, joint_kws={"facecolor": 'white'})#,,  marginal_kws = dict(bins = np.arange(0, 50000))

    for patch in TR.ax_marg_x.patches:
        patch.set_facecolor('grey')

    for patch in TR.ax_marg_y.patches:
        patch.set_facecolor('grey')

    TR.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac) - Train')
    plt.ylabel('Predict (ton/ac)')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(tr_r2, tr_mae, tr_rmse, tr_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Train Data')
    #========================================================
    w_valid_e1  = valid_df[['ytrue', week_pred]]
    w_valid_e1  = w_valid_e1.rename(columns={week_pred: "ypred"})
    valid_true = w_valid_e1['ytrue']
    valid_pred = w_valid_e1['ypred']
    val_r2, val_mae, val_rmse, val_mape, _,_ = regression_metrics(valid_true, valid_pred)

    Va = sns.jointplot(x = valid_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = 'viridis', mincnt = mincnt) #palette ='flare', ,   
    for patch in Va.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Va.ax_marg_y.patches:
        patch.set_facecolor('grey')


    Va.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)
    plt.xlabel('Measured (ton/ac) - Validation')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(val_r2, val_mae, val_rmse, val_mape)
    plt.legend([extra], [scores], loc='upper left')
    #plt.title('Validation Data')
    #========================================================


    w_test_e1  = test_df[['ytrue', week_pred]]
    w_test_e1  = w_test_e1.rename(columns={week_pred: "ypred"})
    test_true = w_test_e1['ytrue']
    valid_pred = w_test_e1['ypred']

    test_r2, test_mae, test_rmse, test_mape, _,_ = regression_metrics(test_true, valid_pred)

    Te = sns.jointplot(x=test_true, y = valid_pred, kind="hex", height=8, ratio=4, 
                        xlim = [0,30], ylim = [0,30], extent=[0, 30, 0, 30], gridsize=100, 
                        cmap = cmap, mincnt = mincnt) #palette ='flare', ,  
    for patch in Te.ax_marg_x.patches:
        patch.set_facecolor('grey')
    for patch in Te.ax_marg_y.patches:
        patch.set_facecolor('grey') 

    Te.ax_joint.plot([0, 30], [0, 30],'--r', linewidth=2)

    plt.xlabel('Measured (ton/ac)')
    plt.ylabel('')
    plt.grid(False)

    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                            edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'MAE={:.2f}' + '\n' + r'$RMSE={:.2f}$'+'\n' + r'$MAPE={:.2f}$').format(test_r2, test_mae, test_rmse, test_mape)
    plt.legend([extra], [scores], loc='upper left')



    fig = plt.figure(figsize=(21, 7))
    gs = gridspec.GridSpec(1, 3)

    mg0 = SeabornFig2Grid(TR, fig, gs[0])
    mg1 = SeabornFig2Grid(Va, fig, gs[1])
    mg2 = SeabornFig2Grid(Te, fig, gs[2])


    gs.tight_layout(fig)
    #gs.update(top=0.7)
    #plt.savefig(fig_save_name, dpi = 300)
    plt.show()



class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def ScenarioEvaluation2D(pred_npy):

    OutDF = pd.DataFrame()
    out_ytrue= []
    out_ypred_w1, out_ypred_w2, out_ypred_w3, out_ypred_w4, out_ypred_w5, out_ypred_w6, out_ypred_w7, out_ypred_w8 = [],[],[],[],[],[],[],[]
    out_ypred_w9, out_ypred_w10, out_ypred_w11, out_ypred_w12, out_ypred_w13, out_ypred_w14, out_ypred_w15 = [],[],[],[],[],[],[]
  
    for l in range(len(pred_npy)):

        block_key = next(iter(pred_npy[l]))
        block_allpatch = pred_npy[l][block_key]        
            
        ytrue     = pred_npy[l]['ytrue']
        ytrue_flat = ytrue.flatten()
        out_ytrue.append(ytrue_flat)
        
        ypred_w1  = pred_npy[l]['ypred_w1']
        ypred_w1_flat = ypred_w1.flatten()
        out_ypred_w1.append(ypred_w1_flat)
        
        ypred_w2  = pred_npy[l]['ypred_w2']
        ypred_w2_flat = ypred_w2.flatten()
        out_ypred_w2.append(ypred_w2_flat)
        
        ypred_w3  = pred_npy[l]['ypred_w3']
        ypred_w3_flat = ypred_w3.flatten()
        out_ypred_w3.append(ypred_w3_flat)
        
        ypred_w4  = pred_npy[l]['ypred_w4']
        ypred_w4_flat = ypred_w4.flatten()
        out_ypred_w4.append(ypred_w4_flat)
        
        ypred_w5  = pred_npy[l]['ypred_w5']
        ypred_w5_flat = ypred_w5.flatten()
        out_ypred_w5.append(ypred_w5_flat)
        
        ypred_w6  = pred_npy[l]['ypred_w6']
        ypred_w6_flat = ypred_w6.flatten()
        out_ypred_w6.append(ypred_w6_flat)
        
        ypred_w7  = pred_npy[l]['ypred_w7']
        ypred_w7_flat = ypred_w7.flatten()
        out_ypred_w7.append(ypred_w7_flat)
        
        ypred_w8  = pred_npy[l]['ypred_w8']
        ypred_w8_flat = ypred_w8.flatten()
        out_ypred_w8.append(ypred_w8_flat)
        
        ypred_w9  = pred_npy[l]['ypred_w9']
        ypred_w9_flat = ypred_w9.flatten()
        out_ypred_w9.append(ypred_w9_flat)
        
        ypred_w10  = pred_npy[l]['ypred_w10']
        ypred_w10_flat = ypred_w10.flatten()
        out_ypred_w10.append(ypred_w10_flat)
        
        ypred_w11  = pred_npy[l]['ypred_w11']
        ypred_w11_flat = ypred_w11.flatten()
        out_ypred_w11.append(ypred_w11_flat)
        
        ypred_w12  = pred_npy[l]['ypred_w12']
        ypred_w12_flat = ypred_w12.flatten()
        out_ypred_w12.append(ypred_w12_flat)
        
        ypred_w13  = pred_npy[l]['ypred_w13']
        ypred_w13_flat = ypred_w13.flatten()
        out_ypred_w13.append(ypred_w13_flat)
        
        ypred_w14  = pred_npy[l]['ypred_w14']
        ypred_w14_flat = ypred_w14.flatten()
        out_ypred_w14.append(ypred_w14_flat)
        
        ypred_w15  = pred_npy[l]['ypred_w15']
        ypred_w15_flat = ypred_w15.flatten()
        out_ypred_w15.append(ypred_w15_flat)
      


            
    out_ytrue = np.concatenate(out_ytrue)#.astype(np.float16)
    out_ypred_w1 = np.concatenate(out_ypred_w1) 
    out_ypred_w2 = np.concatenate(out_ypred_w2) 
    out_ypred_w3 = np.concatenate(out_ypred_w3) 
    out_ypred_w4 = np.concatenate(out_ypred_w4) 
    out_ypred_w5 = np.concatenate(out_ypred_w5) 
    out_ypred_w6 = np.concatenate(out_ypred_w6) 
    out_ypred_w7 = np.concatenate(out_ypred_w7) 
    out_ypred_w8 = np.concatenate(out_ypred_w8) 
    out_ypred_w9 = np.concatenate(out_ypred_w9) 
    out_ypred_w10 = np.concatenate(out_ypred_w10) 
    out_ypred_w11 = np.concatenate(out_ypred_w11) 
    out_ypred_w12 = np.concatenate(out_ypred_w12) 
    out_ypred_w13 = np.concatenate(out_ypred_w13) 
    out_ypred_w14 = np.concatenate(out_ypred_w14) 
    out_ypred_w15 = np.concatenate(out_ypred_w15) 

    
    OutDF['ytrue'] = out_ytrue
    OutDF['ypred_w1'] = out_ypred_w1
    OutDF['ypred_w2'] = out_ypred_w2 
    OutDF['ypred_w3'] = out_ypred_w3 
    OutDF['ypred_w4'] = out_ypred_w4 
    OutDF['ypred_w5'] = out_ypred_w5 
    OutDF['ypred_w6'] = out_ypred_w6 
    OutDF['ypred_w7'] = out_ypred_w7 
    OutDF['ypred_w8'] = out_ypred_w8 
    OutDF['ypred_w9'] = out_ypred_w9 
    OutDF['ypred_w10'] = out_ypred_w10 
    OutDF['ypred_w11'] = out_ypred_w11 
    OutDF['ypred_w12'] = out_ypred_w12 
    OutDF['ypred_w13'] = out_ypred_w13 
    OutDF['ypred_w14'] = out_ypred_w14 
    OutDF['ypred_w15'] = out_ypred_w15


    return OutDF


def return_samples_error_per_bins(df): 

    MAPE_Errors, counts = [], []
    for i in range(30):
        if i == 0:
            Data  = df.loc[(df['ytrue'] >= i) & (df['ytrue'] <= (i+1))]
        else: 
            Data  = df.loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))] 
        counts.append(len(Data))
        if len(Data) == 0: 
            MAPE = 0
        else:
            MAPE = mean_absolute_percentage_error(Data['ytrue'], Data['ypred_w15'])
        if MAPE > 1: 
            MAPE = 1
        MAPE_Errors.append(MAPE*100)

    return counts, MAPE_Errors

def return_samples_weight_per_bins(df): 
    ytrue_counts, weight_means = [], []
    for i in range(30):
            if i == 0:
                    yture_count = df.loc[(df['ytrue'] >= i) & (df['ytrue'] <= (i+1))]
                    weight_mean = df['weights'].loc[(df['ytrue'] >= i) & (df['ytrue'] <= (i+1))].mean()
            
            else: 
                    yture_count = df.loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))] 
                    weight_mean = df['weights'].loc[(df['ytrue'] > i) & (df['ytrue'] <= (i+1))].mean()

            ytrue_counts.append(len(yture_count))
            weight_means.append(weight_mean)

    return weight_means, ytrue_counts