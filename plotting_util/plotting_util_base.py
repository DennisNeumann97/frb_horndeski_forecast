# Loading in packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import getdist
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
from glob import glob
from getdist.types import ResultTable, MargeStats

# Determining plotting style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#print(mpl.rcParams.keys)

class contour_triangle_plot:
    def __init__(self,
                 labels_samples,             # list of list of pretty names
                 names_samples,              # list of list of callable names
                 kind_samples,               # list of 'fisher' or 'mcmc'
                 fiducial,                   # list of list of fiducial values, put empty entries for MCMC
                 mcmc_prior,                 # dictionary of prior range of MCMC
                 path_to_sample,             # list of full path to sample from working dir
                 filename_fisher,            # list of well defined strings because different fisher matrices in a single folder
                 plot_labels,                # Labels for the samples
                 params_to_show,             # params_which should be shown, rest will be marginalized over
                 fontsize=385                # fontsize (must be huge for large contour plots)
                ):
        
        self.fontsize = fontsize
        self.n_sample = len(labels_samples[0])
        self.n_contours = len(labels_samples)
        self.plot_labels = plot_labels
        self.params_to_show = params_to_show
        self.n_params_to_plot = len(params_to_show)
        self.names_samples = names_samples
        
        def find_entry(lst, target_string):
            for element in lst:
                if target_string in element:
                    return element
            return None  # Return None if no matching entry is found
        
        self.sample_instances = []
        # Loading in points
        for idx, kind in enumerate(kind_samples):
            if kind == 'mcmc':
                datafiles = glob(path_to_sample[idx]+'/*')
                mcmc_points = np.load(find_entry(datafiles,'points'))
                mcmc_weights = np.exp(np.load(find_entry(datafiles,'weights')))
                mcmc_log_likelihoods = -np.load(find_entry(datafiles,'log_likelihood'))
                
                self.sample_instances.append(MCSamples(samples=mcmc_points, weights=mcmc_weights, 
                                                  loglikes=mcmc_log_likelihoods,names=names_samples[idx], 
                                                  labels=labels_samples[idx], ranges=mcmc_prior, sampler='nested'))
                
            elif kind =='fisher':
                datafile = path_to_sample[idx] + filename_fisher[idx]
                fisher_matrix = np.array(np.loadtxt(datafile))
                gaussian_from_fisher = GaussianND(mean=fiducial[idx], cov=fisher_matrix, 
                                  is_inv_cov=True, names=names_samples[idx], labels=labels_samples[idx])
                self.sample_instances.append(gaussian_from_fisher.MCSamples(size=100000))
                
#             # For conditionalised
#             elif kind =='fisher':
#                 indices =[0,2,5]
#                 datafile = path_to_sample[idx] + filename_fisher[idx]
#                 fisher_matrix = np.array(np.loadtxt(datafile))
#                 gaussian_from_fisher = GaussianND(mean=[fiducial[idx][i] for i in indices],
#                                                   cov=[[fisher_matrix[i,j] for j in indices] for i in indices], 
#                                                   is_inv_cov=True, 
#                                                   names=[names_samples[idx][i] for i in indices],
#                                                   labels=[labels_samples[idx][i] for i in indices])
#                 self.sample_instances.append(gaussian_from_fisher.MCSamples(size=100000))
                
    def get_plot(self, figurename, fiducial_dict, suptitle: str=None):
        colors = plt.cm.viridis(np.linspace(0, 0.75, self.n_contours))
        cmap = mpl.colors.ListedColormap([color for color in colors][::-1])

        plt.rcParams.update({'font.size': self.fontsize})
        g = plots.get_subplot_plotter()
        g.settings.axis_tick_max_labels = 4
        g.settings.axis_tick_step_groups = [[1,2,3,4,5,6,7,8,9,10]]
        g.settings.solid_colors = cmap
        g.settings.tight_layout = True
        g.settings.axes_fontsize = 12
        g.settings.legend_fontsize = 17
        g.triangle_plot(self.sample_instances, filled=True, legend_labels=self.plot_labels,
                        markers=fiducial_dict, params=self.params_to_show)
#         g.triangle_plot(self.sample_instances, filled=True, legend_labels=self.plot_labels,
#                         params=self.params_to_show)
        for i in range(self.n_params_to_plot):
            g.rotate_xticklabels(ax=[self.n_params_to_plot-1,i], rotation=45)
            g.rotate_yticklabels(ax=[i,0], rotation=45)

        if suptitle is not None:
            plt.suptitle(suptitle, fontsize=20, y=0.735)
        plt.savefig('./contour_plots/'+figurename, bbox_inches='tight', dpi=300)
        plt.show()
        
    def get_single_tile_plot(self, figurename, fiducial_dict, param1, param2):
        colors = plt.cm.viridis(np.linspace(0, 0.75, self.n_contours))
        cmap = mpl.colors.ListedColormap([color for color in colors][::-1])

        plt.rcParams.update({'font.size': self.fontsize})
        g = plots.get_subplot_plotter()
        g = plots.get_subplot_plotter()
        g.settings.axis_tick_max_labels = 4
        g.settings.axis_tick_step_groups = [[1,2,3,4,5,6,7,8,9,10]]
        g.settings.solid_colors = cmap
        g.settings.tight_layout = True
        g.plot_2d(self.sample_instances, param1, param2, filled=True, legend_labels=self.plot_labels,
                        markers=fiducial_dict)
        axes = g.get_axes()
        g.rotate_xticklabels(rotation=45)
        g.rotate_yticklabels(rotation=45)
        
        plt.savefig('./contour_plots/'+figurename, bbox_inches='tight', dpi=300)
        plt.show()
        
    def get_latex_table(self, n_sigma):
        print(ResultTable(ncol=1,results=self.sample_instances, limit=n_sigma).tableTex())
        
    def get_upper_bounds(self, instance_idx, param_name):
        mean = self.sample_instances[instance_idx].getMargeStats().parsWithNames(param_name)[0].mean
        upper = self.sample_instances[instance_idx].getMargeStats().parsWithNames(param_name)[0].limits[0].upper
        return upper-mean
    
    def get_uncertainty_range(self, instance_idx, param_name):
        mean = self.sample_instances[instance_idx].getMargeStats().parsWithNames(param_name)[0].mean
        lower = self.sample_instances[instance_idx].getMargeStats().parsWithNames(param_name)[0].limits[0].lower
        upper = self.sample_instances[instance_idx].getMargeStats().parsWithNames(param_name)[0].limits[0].upper
        return (upper-lower)
    
    def print_improvement_stats(self, instance_pairs):
        for pair in instance_pairs:

            improvement_list = []
            for name in self.names_samples[pair[0]]:
                sigma_range1 = self.get_uncertainty_range(pair[0], name)
                sigma_range2 = self.get_uncertainty_range(pair[1], name)
                improvement_list.append(sigma_range1/sigma_range2)

            improvement_dict = dict(zip(self.names_samples[pair[0]], improvement_list))

            print("Improvement for instance pair: ", pair)
            print(improvement_dict)