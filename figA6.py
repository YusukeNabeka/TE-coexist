#figA6
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def plot_ld_dynamics_with_mean(condition_list, data_dir, downsample_steps):
    df_all_ls=[]
    df_mean_ls=[]
    for condition_prefix, downsample_step in zip(condition_list,downsample_steps):
        search_pattern = os.path.join(data_dir, f"*{condition_prefix}*rep*.tsv")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"No files found for pattern: {search_pattern}")
            return
            
        print(f"Found {len(files)} files. Processing...")
        
        df_list = []
        for i, f in enumerate(files):
            df = pd.read_csv(f, sep='\t')
            
            if downsample_step > 1:
                df = df[df['gen'] % downsample_step == 0]
                
            df['rep_id'] = f"rep_{i}"
            df_list.append(df)
            
        df_all = pd.concat(df_list, ignore_index=True)
        
        df_mean = df_all.groupby('gen').mean().reset_index()

        df_all_ls.append(df_all)
        df_mean_ls.append(df_mean)

        
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(13.3, 10.5), constrained_layout=False)
    mosaic = """
        147
        258
        369
    """
    ax_dict = fig.subplot_mosaic(mosaic)
    ax123 = [ax_dict["1"], ax_dict["2"], ax_dict["3"]]
    ax456 = [ax_dict["4"], ax_dict["5"], ax_dict["6"]]
    ax789 = [ax_dict["7"], ax_dict["8"], ax_dict["9"]]

    ax_dict["1"].sharey(ax_dict["4"])


    def plot_var(ax, col, color, label, df_all, df_mean):
        sns.lineplot(data=df_all, x='gen', y=col, units='rep_id', estimator=None, 
                     color=color, alpha=0.075, linewidth=0.7, ax=ax, legend=False)
        ax.plot(df_mean['gen'], df_mean[col], '-', lw=3, color=color, label=label)
        ax.grid(True, ls=":")
        ax.set_xlabel("")
        
        ax.tick_params("y", labelsize=15)
        ax.tick_params("x", labelsize=15)

        


    for axes, dfa, dfm in zip([ax123, ax456, ax789],df_all_ls,df_mean_ls):
        plot_var(axes[0], 'te', 'red', 'TE mean', dfa, dfm)
        axes[0].set_ylabel('TE copy number', fontsize=18, labelpad=14)
        axes[0].set_ylim(-2,42)

        plot_var(axes[1], 'kz_freq', 'green', 'KZFP mean', dfa, dfm)
        plot_var(axes[1], 'pi_freq', 'blue', 'piRNA mean', dfa, dfm)
        axes[1].set_ylabel('Allele frequency', fontsize=18, labelpad=10)
        axes[1].set_ylim(-0.05, 1.05)

        axes[2].axhline(0, color='k', linestyle='--', linewidth=1)
        plot_var(axes[2], 'cov_kz_te', 'xkcd:puke green', 'KZFP–TE mean', dfa, dfm)
        plot_var(axes[2], 'cov_pi_te', 'tab:cyan', 'piRNA–TE mean', dfa, dfm)
        axes[2].set_ylabel('Covariance', fontsize=18, labelpad=2)
        axes[2].set_xlabel("Generations", fontsize=18)
        axes[2].set_ylim(-0.125, 0.075)
        
        
        for i in ax123:
            i.set_xlim(-0,2000)
        for i in ax456:
            i.set_xlim(-0,35000)
        for i in ax789:
            i.set_xlim(-0,15000)

    ax789[0].legend(loc='upper right', fontsize=14, frameon=False)
    ax789[1].legend(loc='upper right', fontsize=14, frameon=False)
    ax789[2].legend(loc='upper right', fontsize=14, frameon=False)

    ax_dict["1"].set_title("KZFP loss", fontsize=18, pad=28)
    ax_dict["1"].text(
        0.5, 1.01,
        r"$f=0.40$, $c=3\times10^{-3}$, $r=0.499$",
        transform=ax_dict["1"].transAxes,
        ha="center", va="bottom",
        fontsize=14
    )

    ax_dict["4"].set_title("KZFP fixation", fontsize=18, pad=28)
    ax_dict["4"].text(
        0.5, 1.01,
        r"$f=0.40$, $c=3\times10^{-5}$, $r=0.499$",
        transform=ax_dict["4"].transAxes,
        ha="center", va="bottom",
        fontsize=14
    )

    ax_dict["7"].set_title("KZFP polymorphism", fontsize=18, pad=28)
    ax_dict["7"].text(
        0.5, 1.01,
        r"$f=0.80$, $c=3\times10^{-5}$, $r=0.499$",
        transform=ax_dict["7"].transAxes,
        ha="center", va="bottom",
        fontsize=14
    )

    fig.text(0.025, 0.98, "A", fontsize=25, color="k", fontweight="bold")
    fig.text(0.35, 0.98, "B", fontsize=25, color="k", fontweight="bold")
    fig.text(0.675, 0.98, "C", fontsize=25, color="k", fontweight="bold")

    plt.tight_layout()
    
    plt.savefig('figA6.pdf', format="pdf", bbox_inches='tight')
    plt.show()
    



if __name__ == "__main__":
    data_directory = "./simulation/data/phase2/*/*"
    
    target_condition1 = "r0.499_rep*fkz0.40_ckz3.0e-03"
    target_condition2 = "r0.499_rep*fkz0.40_ckz3.0e-05"
    target_condition3 = "r0.499_rep*fkz0.80_ckz3.0e-05"
    
    target_conditions = [target_condition1,target_condition2,target_condition3]
    steps = [100,1800,700]

    plot_ld_dynamics_with_mean(target_conditions, data_directory, steps)
    

