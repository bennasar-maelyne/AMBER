import numpy as np
import scipy.sparse as sparse
import random
import pandas as pd
from scipy.io import loadmat
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def sigmoid(L, x, x0, k): #sigmoid function
    return L/(1 + np.exp(-k*(x-x0)))

class Voxel(object): #extra parameters are max_occupancy, viscosity
        def __init__(self, position, half_length, viscosity, list_of_cells_in=None, n_capillaries = 0, voxel_number = 0):
                if list_of_cells_in is None:
                        list_of_cells_in = np.array([])
                self.position = position #position is a 3D vector
                self.half_length = half_length #half_length is a scalar
                self.list_of_cells = list_of_cells_in #list of alive cells in the voxel
                self.list_of_dead_cells = [] #list of necrotic cells in the voxel
                self.n_capillaries = n_capillaries #number of capillaries in the voxel
                self.volume = 8*half_length**3 #volume of the voxel
                self.voxel_number = voxel_number #id of the voxel
                self.dose = 0 #dose in the voxel from last irradiation
                self.molecular_factors = {'VEGF': 0} #molecular factors in the voxel. Only VEGF is implemented
                self.viscosity = viscosity #viscosity of the voxel
                self.vessel_volume = 0 #volume of the vessels in the voxel
                self.vessel_length = 0 #length of the vessels in the voxel
                self.bifurcation_density = 0 #bifurcation density in the voxel
                self.pH = 7.4 #pH in the voxel

        def number_of_tumor_cells(self): #returns the number of tumor cells in the voxel
                number = 0
                for cell in self.list_of_cells:
                        if cell.type == 'TumorCell':
                                number = number + 1
                return number

        def number_of_dead_cells(self): #returns the number of dead (necrotic+apoptotic) cells in the voxel
                return len(self.list_of_dead_cells)

        def number_of_necrotic_cells(self): #returns the number of apoptotic cells in the voxel
                number = 0
                for cell in self.list_of_dead_cells:
                        if cell.necrotic == True:
                                number = number + 1
                return number

        def number_of_apoptotic_cells(self): #returns the number of apoptotic cells in the voxel
                number = 0
                for cell in self.list_of_dead_cells:
                        if cell.necrotic == False:
                                number = number + 1
                return number

        def number_of_alive_cells(self): #returns the number of alive cells in the voxel
                return len(self.list_of_cells)
        
        def occupied_volume(self): #returns the volume occupied by the cells in the voxel
                volume = 0.0
                for cell in self.list_of_cells:
                        volume = volume + cell.volume
                for cell in self.list_of_dead_cells:
                        volume = volume + cell.volume
                return volume
        
        def occupied_volume_fraction(self): #returns the volume fraction occupied by the cells in the voxel
                return self.occupied_volume() / self.volume

        def vessel_volume_density(self): #returns the vessel volume density in the voxel
                side = 2*self.half_length
                capillary_volume = side * np.pi * 0.002 ** 2
                vessel_volume_density = ((self.vessel_volume + self.n_capillaries * capillary_volume) / self.volume)*100
                return vessel_volume_density
        
        def vessel_length_density(self): #returns the vessel length density in the voxel
                side = 2*self.half_length
                vessel_length_density = (self.vessel_length + self.n_capillaries * side) / self.volume
                return vessel_length_density

        def pressure(self): #returns the pressure in the voxel
                packing_density = (self.occupied_volume() /self.volume)
                return packing_density

        def random_points_in_voxel(self, n): #returns n random points in the voxel
                points = np.random.uniform(-self.half_length, self.half_length, (n,3))
                points = points + self.position
                return points
        
        def add_cell(self, cell, max_occupancy): #try to add a cell to the voxel
                if self.pressure() > max_occupancy:
                        #print('Voxel is full, pressure is', self.pressure(), ' number of cells is', self.number_of_alive_cells(), ' and number of necrotic cells is', self.number_of_necrotic_cells())
                        return False
                else:
                        self.list_of_cells = np.append(cell, self.list_of_cells)
                        return True

        def remove_cell(self, cell): #remove a cell from the voxel
                id = np.where(self.list_of_cells == cell)
                self.list_of_cells = np.delete(self.list_of_cells, id)
                return True

        def remove_dead_cell(self, cell):
                indices = np.nonzero(self.list_of_dead_cells == cell)[0]
                if indices.size == 0:
                        return False  # Cell not found in the list
                self.list_of_dead_cells = np.delete(self.list_of_dead_cells, indices)
                return True

        def cell_becomes_necrotic(self, cell): #remove a cell from the voxel and add it to the list of necrotic cells
                self.list_of_cells = np.delete(self.list_of_cells, np.where(self.list_of_cells == cell))
                cell.necrotic = True
                self.list_of_dead_cells = np.append(cell, self.list_of_dead_cells)
                return True

        def cell_becomes_apoptotic(self, cell):
                self.list_of_cells = np.delete(self.list_of_cells, np.where(self.list_of_cells == cell))
                self.list_of_dead_cells = np.append(cell, self.list_of_dead_cells)
                return True

        def oxygen_histogram(self, ax, fig): #plot the oxygen histogram of the voxel
                oxygen = []
                for cell in self.list_of_cells:
                        oxygen = np.append(oxygen, cell.capillaries)
                ax.hist(oxygen, bins = 50, color = 'blue', alpha = 0.5, range = (0,1))
                ax.set_xlim(0, 1)
                ax.set_ylabel('Number of cells')
                ax.set_title('Oxygen histogram')
                return ax, fig

        def vitality_histogram(self, ax, fig): #plot the vitality histogram of the voxel
                vitality = []
                for cell in self.list_of_cells:
                        vitality.append(cell.vitality())
                ax.hist(vitality, bins=50, color='orange', alpha=0.5, range=(0, 1))
                ax.set_xlim(0, 1)
                ax.set_xlabel(f'Vitality, N_capillaries in voxel was = {self.n_capillaries}, nb of cells = {self.number_of_alive_cells()}')
                ax.set_ylabel('Number of cells')
                ax.set_title('Vitality histogram')
                return ax, fig

        def cycling_time_and_age_histogram(self, ax, fig): #plot the cycling time and age histogram of the voxel

                if self.number_of_tumor_cells() == 0:
                        return ax, fig

                from scipy.stats import gamma
                gamma_scale = self.list_of_cells[0].gamma_scale
                gamma_shape = self.list_of_cells[0].gamma_shape
                cycling_time = []
                age = []
                for cell in self.list_of_cells:
                        cycling_time.append(cell.doubling_time)
                        age.append(cell.time_spent_cycling)
                ax.hist(cycling_time, bins=20, color='green', alpha=0.5, label='Time before next division', density=True)
                x = np.linspace(0, 70, 100)
                ax.hist(age, bins=20, color='red', alpha=0.5, label='Time spent cycling', density=True)
                ax.plot(x, gamma.pdf(x, gamma_shape, scale=gamma_scale), linestyle = 'dashed', color = 'black', lw=4, alpha=1.0, label='Gamma distribution')
                ax.set_xlabel('Time [h]')
                ax.set_ylabel('Frequency')
                ax.set_title('Cycling time histogram')
                ax.legend()
                return ax, fig

        def compute_cell_interaction_matrix(self, dt): #compute the cell interaction matrix in the voxel
                cell_interaction_matrix = sparse.lil_matrix((len(self.list_of_cells), len(self.list_of_cells)), dtype=np.float32)
                for i in range(len(self.list_of_cells)):
                        for j in range(len(self.list_of_cells)):
                                if i != j:
                                        cell_interaction_matrix[i,j] = self.list_of_cells[i].probability_of_interaction(self.list_of_cells[j], dt)
                cell_interaction_matrix = cell_interaction_matrix.tocsr()
                return cell_interaction_matrix

        def average_cell_damage(self): #returns the average damage of the cells in the voxel
                if len(self.list_of_cells) == 0:
                        return -1
                damage = 0
                for cell in self.list_of_cells:
                        damage = damage + cell.damage
                return damage / len(self.list_of_cells)
        
        #needs to be improved by taking into account density and CT intensity
        def biological_to_HU(self, vitality_cycling_threshold): #compute the mean HU value of the voxel

                N_dead=self.number_of_dead_cells()
                N_living=0
                for cell in self.list_of_cells:
                        N_living+=1
                f_vessels=self.vessel_volume_density()/100 #fraction of vessels occupation in the voxel
                N_tot=N_dead+N_living
                voxel_volume=self.volume
                

                #Cell density (nb cells/mm**3)
                rho_living=N_living/voxel_volume
                rho_dead=N_dead/voxel_volume
                rho_cells=rho_living+rho_dead

                #Cellular volume fraction
                if N_tot==0:
                        f_living=0
                        f_dead=0
                else:
                        cell_volume= 4/3 * np.pi * 0.033**3  #adapt the radius of the cell
                        f_living=N_living*cell_volume/voxel_volume
                        f_dead=N_dead*cell_volume/voxel_volume
                
                #rest of the voxel is considered as a fluid interstitial
                f_extracellular=1-f_vessels-f_living-f_dead
                if f_extracellular<0.5:
                        print('f_living=',f_living,'f_dead=',f_dead,'f_vessels=',f_vessels)
                        print('At least half of the volume of the voxel is occupied by cells and vessels in',self.voxel_number)

                def density_effect(rho_cells, rho_ref, cell_type="living"):
                        if rho_cells == 0:
                                return 0
                        density_ratio = rho_cells / rho_ref
        
                        if cell_type == "dead": #Negative effect because dead cells are less dense
                        #        print('Density effect of dead cells',-10 * np.log10(1 + density_ratio))
                                return -10 * np.log10(1 + density_ratio)  
                        else: # Positive effect for alive cells
                        #        print('Density effect of living cells',20 * np.log10(1 + density_ratio))
                                return 20 * np.log10(1 + density_ratio)
                
                #HU values need to be adapted or estimated with real data : here it is liver range HU values
                HU_living = random.randint(115,135) # + density_effect(rho_living,1e3,"living") adapt rho_ref according to the parameters of the simulation
                HU_dead =  random.randint(20,35) # + density_effect(rho_dead,1e3,"dead")
                HU_vessels = random.randint(30, 45)
                HU_extracellular = random.randint(64,76)  

                return f_living*HU_living+f_dead*HU_dead+f_vessels*HU_vessels+f_extracellular*HU_extracellular
        
        def MRI_voxel_intensity(self, vitality_threshold, MRI_sequence, T1_base, T2_base, PD_base, TE, TR, TI):
                N_dead = self.number_of_dead_cells()
                N_living = len(self.list_of_cells)
                N_quiescent = 0
                for cell in self.list_of_cells:
                        if cell.vitality() < vitality_threshold:
                                N_quiescent+=1
                f_vessels = self.vessel_volume_density() / 100
                N_tot = N_dead + N_living
                voxel_volume = self.volume

                if N_tot == 0:
                        f_living = 0
                        f_dead = 0
                        f_quiescent = 0
                else:
                        cell_volume = 4/3 * np.pi * 0.033**3  # Cell radius in mm
                        f_living = N_living * cell_volume / voxel_volume
                        f_dead = N_dead * cell_volume / voxel_volume
                        f_quiescent = N_quiescent * cell_volume / voxel_volume

                #Coefficients that need to be calibrated on real data
                alpha_1 = 500
                beta_1, beta_2 = 40, 100
                gamma_1, gamma_2 = 0.3, 0.4

                #Generating T1, T2 and rho maps using AMBER outputs
                T1_voxel = T1_base + alpha_1*f_dead
                T2_voxel = T2_base - beta_1*(f_living+f_dead) + beta_2*f_dead
                PD_voxel = PD_base - gamma_1*(f_living+f_dead) + gamma_2*f_dead

                #Make sure that T1, T2 and rho are in physiological range
                T1_voxel = np.clip(T1_voxel, 400, 3000)
                T2_voxel = np.clip(T2_voxel, 20, 300)
                PD_voxel = np.clip(PD_voxel, 0.3, 1.2)

                if MRI_sequence == "T1w_SE" or MRI_sequence == "PDw" :
                        return PD_voxel * (1 - np.exp(-TR / T1_voxel)) * np.exp(-TE / T2_voxel)

                elif MRI_sequence == "T2w_SE":
                        return PD_voxel * np.exp(-TE / T2_voxel)
                
                elif MRI_sequence == "T1_TI":
                        return PD_voxel * (1 - 2*np.exp(-TI/T1_voxel) + np.exp(-TR/T1_voxel)) * np.exp(-TE/T2_voxel)

                else:
                        print(f"MRI_sequence '{MRI_sequence}' has not been implemented.")
                        return -1

        def DWI_MRI_intensity(self, bvalue, bvecs, TD, pulsew, Din, Dex, kappa):
                if len(self.list_of_cells) == 0:
                        return 0.5  #relative intensity of DWI in healthy tissues

                # Getting AMBER outputs
                f = self.occupied_volume_fraction()
                if f<0.007:  #less than 10 cells in the voxel
                        return 0.5
                #print('cellular density f=',f)
        
                radius_list = []
                N_cell=0
                for cell in self.list_of_cells:
                        radius_list.append(cell.radius)
                        N_cell+=1
                rmean = np.mean(radius_list)*1000   #radius is in µm in the look-up table
                rsd = np.std(radius_list)*1000 + 1
                #print('mean radius is ',rmean,' and radius standard deviation ',rsd)

                query_params = np.array([f, Dex, rmean, rsd]) 

                # Loading the look up table (for  2 populations (living and dead) load lookup_table_two_pop.mat and adapt code)
                lut = loadmat('lookup_table.mat')
                params = lut['params']              
                signals = lut['signals_4D']            
                sequence = lut['sequence']

                #If some parameters are not varying, they must be removed from the table to allow the interpolation : here kappa = 0.01
                dim_to_remove = 4
                params = np.delete(params, dim_to_remove, axis=1)
                
                #Extract b values, diffusion time, directions of gradient
                n_bval = int(sequence['n_bval'][0][0][0][0])
                n_bvec = int(sequence['n_bvec'][0][0][0][0])
                n_TD = int(sequence['n_del'][0][0][0][0])
                
                bvals = sequence['bval'][0][0]   # shape: (n_bval,)
                TDs = sequence['TD'][0][0] # shape: (n_del,)
                bvecs = [int(i.strip()) for i in bvecs.split(",")]
                TD_index = np.where(TDs == TD)[0][0]
                if TD_index.size == 0:
                        raise ValueError(f"TD = {TD} not found in the LUT.")
                bval_index = np.where(bvals == bvalue)[0][0]
                if bval_index.size == 0:
                        raise ValueError(f"bval = {bvalue} not found in the LUT.")

                #Create a triangulation is parameters space
                tri = Delaunay(params)

                #Find the simplex containing the point of interest
                simplex_index = tri.find_simplex(query_params)

                if simplex_index == -1:
                        print(f"The point {query_params} is out of the convex hull for triangulation.")
                        print('Minimum of the hull is ', np.min(params, axis=0), ' and maximum is ', np.max(params, axis=0))

                        # Remove Dex
                        params_3D = np.delete(params, 1, axis=1)
                        query_3D = np.delete(query_params, 1)

                        # Convex hull and visualization
                        hull = ConvexHull(params_3D)
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')

                        # LUT points
                        ax.scatter(params_3D[:, 0], params_3D[:, 1], params_3D[:, 2], alpha=0.3, label='LUT points')

                        # Query point
                        ax.scatter(query_3D[0], query_3D[1], query_3D[2], c='r', label='Query point', s=50)

                        # Surface plot of the convex hull
                        faces = [params_3D[simplex] for simplex in hull.simplices]
                        poly3d = Poly3DCollection(faces, facecolors='lightblue', linewidths=0.2, edgecolors='k', alpha=0.3)
                        ax.add_collection3d(poly3d)

                        ax.set_xlabel("f")
                        ax.set_ylabel("rmean")
                        ax.set_zlabel("rsd")
                        ax.legend()
                        plt.tight_layout()
                        plt.show()

                        return np.nan

                else:
                        #print('Minimum of the hull is ', np.min(params, axis=0), ' and maximum is ', np.max(params, axis=0))
                        # Get the vertices of the simplex
                        vertex_index = tri.simplices[simplex_index]  
                        #print(vertex_index)

                        # Extract the coordinates
                        vertices = params[vertex_index]  
                        #print(vertices)

                        # Extract barycentric coordinates
                        T = vertices[1:] - vertices[0]  
                        v = query_params - vertices[0]  
                        bary_coords = np.linalg.solve(T.T, v)  
                        bary_coords = np.append(1 - np.sum(bary_coords), bary_coords) 

                        # Security
                        if np.any(bary_coords < -1e-6):
                                print(f"Negative barycentric coordinates => possible extrapolation")
    
                        # Get the signals for each vertex
                        signals_subset = signals[vertex_index] 

                        # Linear combination of the signals with the weights from the simplex
                        interpolated_signal = np.tensordot(bary_coords, signals_subset, axes=(0, 0))  

                        # Extraction of the signal for TD, bvalue and mean over the multiple bvecs
                        signal_values = interpolated_signal[TD_index, bval_index, :]
                        signal_values = signal_values[bvecs]
                        signal_avg = np.mean(signal_values)/1e5
                        #print(signal_avg)

                #print(signal_avg, 'for the voxel number ', self.voxel_number)
                #print("voxel", self.voxel_number)
                #print("nb cells:", N_cell)
                #print("query params:", query_params)
                #print("TD_index:", TD_index, "bval_index:", bval_index, "bvec_index:", bvecs)

                return signal_avg











