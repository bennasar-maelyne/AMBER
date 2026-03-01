from amber.world import *
from amber.voxel import *
from amber.ScalarField import *
import amber.terminal as term
import amber.ReadAndWrite as rw
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import os
from time import time as current_time
import math
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import datetime

class Simulator: #this class is used to run the whole simulation

    def __init__(self, config, list_of_process : list, finish_time, dt):
        self.list_of_process = list_of_process
        self.list_of_process_names = [process.name for process in list_of_process]
        self.finish_time = finish_time
        self.dt = dt
        self.time = 0
        self.config = config

        if not os.path.exists('DataOutput/'):
            os.makedirs('DataOutput/')

        if not os.path.exists('Plots/'):
            os.makedirs('Plots/')

    def center_of_mass(self, center_of_mass, times): #3D plot of the center of mass
        #3D plot of the center of mass
        #set dpi to 300 for high quality
        size = self.config.half_length_world
        center_of_mass = np.array(center_of_mass)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(center_of_mass[:, 0], center_of_mass[:, 1], center_of_mass[:, 2], 'black')
        ax.scatter(center_of_mass[0, 0], center_of_mass[0, 1], center_of_mass[0, 2], color ='green', label='start')
        ax.scatter(center_of_mass[-1, 0], center_of_mass[-1, 1], center_of_mass[-1, 2], color ='red', label='end')
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size)
        ax.set_zlim(-size, size)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Center of mass path')
        ax.legend()
        plt.savefig('Plots/Center_of_mass.png', dpi=100)
        if self.config.running_on_cluster: #if running on cluster, save plot to file and do not show
            plt.close()
        else:
            plt.show()

    def show_cell_and_tumor_volume(self, number_tumor_cells, number_necrotic_cells, number_quiescent_cells, number_cycling_cells, tumor_size, tumor_size_free, number_vessels, times): #plot the number of cells and tumor volume
        # plot number of cells evolution
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        #change font size
        plt.rcParams.update({'font.size': 14})
        
        
        # Plot number of cells evolution
        ax1.plot(times, number_tumor_cells, 'blue', label='All cells')
        ax1.plot(times, number_cycling_cells, 'red', label='Cycling cells')
        ax1.plot(times, number_quiescent_cells, 'green', label='Quiescent cells')
        ax1.plot(times, number_necrotic_cells, 'black', label='Necrotic cells')
        ax1.set_title('Number of cells evolution')
        ax1.set_xlabel('Time [h]', fontsize=14)
        ax1.set_ylabel('Number of cells', fontsize=14)
        ax1.grid(True)
        ax1.legend(fontsize=14)

        # Plot tumor size evolution
        ax2.plot(times, tumor_size, 'red')
        ax2.plot(times, tumor_size_free, 'blue')
        ax2.set_title('Tumor volume evolution', fontsize=14)
        ax2.set_xlabel('Time [h]', fontsize=14)
        ax2.set_ylabel('Tumor volume [mm^3]', fontsize=14)
        ax2.grid(True)

        # Plot number of vessels evolution
        ax3.plot(times, number_vessels, 'black')
        ax3.set_title('Number of vessels evolution', fontsize=14)
        ax3.set_xlabel('Time [h]', fontsize=14)
        ax3.set_ylabel('Number of vessels', fontsize=14)
        ax3.grid(True)

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Save the figure
        fig.savefig('Plots/Combined_plots_tumor_evolution.png', dpi=100)
        if self.config.running_on_cluster:
            plt.close()
        else:
            plt.show()

    def show(self, world: World, t = 0): #this function is used to show the world at a certain time
        print('Showing world at time : ', t)
        start = current_time()

        if not os.path.exists('Plots/CurrentPlotting/'):
            os.makedirs('Plots/CurrentPlotting/')

        size = world.half_length
        print('World half length is',world.half_length)

        if self.config.show_angiogenesis_metrics: #if angiogenesis metrics are to be shown, show them
            print('Showing angiogenesis metrics')
            world.show_angiogenesis_metrics(t, self.config.true_vasculature)

            #
            # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 20))
            # axes[0].set_xlim(-size, size)
            # axes[0].set_ylim(-size, size)
            # world.show_tumor_slice(axes[0], fig, 'vessel_length_density', levels= np.linspace(0, 200, 20), refinement_level=3, cmap='jet')
            # axes[0].grid(True)
            # axes[0].set_facecolor('whitesmoke')
            # axes[0].set_title('Vessel length density [mm/mm^3]')
            #
            # axes[1].set_xlim(-size, size)
            # axes[1].set_ylim(-size, size)
            # world.show_tumor_slice(axes[1], fig, 'vessel_volume_density', refinement_level=3, cmap='jet')
            # axes[1].grid(True)
            # axes[1].set_facecolor('whitesmoke')
            # axes[1].set_title('Vessel volume density [%]')
            #
            # plt.tight_layout()
            # plt.savefig('Plots/CurrentPlotting/t' + str(t) + '_VesselMetricsMaps.png', dpi=100)
            # if self.config.running_on_cluster:
            #     plt.close()
            # else:
            #     plt.show()


        #plot vasculature
        if self.config.show_tumor_and_vessels_3D:
            print('Showing tumor and vessels 3D')
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), subplot_kw={'projection': '3d'})
            plt.rcParams.update({'font.size': 10})
            plt.title('Visualization at time t = ' + str(t) + ' hours', fontsize=16)
            axes.set_xlim(-size, size)
            axes.set_ylim(-size, size)
            axes.set_zlim(-size, size)
            #remove z axis ticks
            if self.config.slice == 'x':
                axes.set_zticks([])
            #change text size
            # axes.view_init(90, -90)
            # world.show_tumor_3D(axes, fig, 'number_of_tumor_cells', cmap='viridis', vmin=0, vmax=1000)
            world.vasculature.plot(fig, axes)
            plt.tight_layout()
            plt.savefig('Plots/CurrentPlotting/t' + str(t) + '_Vasculature.png', dpi=300)
            if self.config.running_on_cluster:
                plt.close()
            else:
                plt.show()

        if self.config.show_slices:
            font = 22
            print('Showing slices')
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
            fig.suptitle('t = ' + str(t) + 'h', fontsize=16)
            voxel_side = 2*self.config.half_length_world / self.config.voxel_per_side

            axes[0, 0].set_xlim(-size, size)
            axes[0, 0].set_ylim(-size, size)
            world.show_tumor_slice(axes[0, 0], fig, 'number_of_alive_cells', levels= np.linspace(1, 1000, 11), cmap='viridis', extend = 'neither')
            #plot the three voxels that are used for o2 histograms
            axes[0, 0].plot(0, 0, 'bo', label='Point')
            axes[0, 0].annotate(f'({0}, {0})', xy=(0, 0), xytext=(3, 3), textcoords='offset points', fontsize=9, color='black')
            axes[0, 0].plot(3*voxel_side, 3*voxel_side, 'bo', label='Point')
            axes[0, 0].annotate(f'({round(3*voxel_side,1)}, {round(3*voxel_side,1)})', xy=(3*voxel_side, 3*voxel_side), xytext=(3, 3), textcoords='offset points', fontsize=9, color='black')
            axes[0, 0].plot(1.5*voxel_side, 1.5*voxel_side, 'bo', label='Point')
            axes[0, 0].annotate(f'({round(1.5*voxel_side,1)}, {round(1.5*voxel_side,1)})', xy=(1.5*voxel_side, 1.5*voxel_side), xytext=(3, 3), textcoords='offset points', fontsize=9, color='black')

            axes[0,0].grid(True)
            axes[0,0].set_facecolor('whitesmoke')
            axes[0, 0].set_title('Number of Alive Tumor Cells', fontsize=font)

            norm = TwoSlopeNorm(vmin=0, vcenter=20, vmax=110)
            #norm = Normalize(vmin=0, vmax=110)

            axes[0, 1].set_xlim(-size, size)
            axes[0, 1].set_ylim(-size, size)
            world.show_tumor_slice(axes[0, 1], fig, 'n_capillaries', cmap = 'RdBu', norm = norm, levels= np.linspace(0, 50, 11), round_n = 0)
            #plot the three voxels that are used for o2 histograms
            axes[0, 1].plot(0, 0, 'bo', label='Point')
            axes[0, 1].annotate(f'({0}, {0})', xy=(0, 0), xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')
            axes[0, 1].plot(3*voxel_side, 3*voxel_side, 'bo', label='Point')
            axes[0, 1].annotate(f'({round(3*voxel_side,1)}, {round(3*voxel_side,1)})', xy=(3*voxel_side, 3*voxel_side), xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')
            axes[0, 1].plot(1.5*voxel_side, 1.5*voxel_side, 'bo', label='Point')
            axes[0, 1].annotate(f'({round(1.5*voxel_side,1)}, {round(1.5*voxel_side,1)})', xy=(1.5*voxel_side, 1.5*voxel_side), xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')
            axes[0, 1].grid(True)
            axes[0, 1].set_facecolor('whitesmoke')
            axes[0, 1].set_title('Number of Capillaries', fontsize=font)

            axes[1, 0].set_xlim(-size, size)
            axes[1, 0].set_ylim(-size, size)
            world.show_tumor_slice(axes[1, 0], fig, 'molecular_factors', factor='VEGF', levels= np.linspace(0.001, 1, 11), cmap='Oranges', round_n = 2)
            #plot the three voxels that are used for o2 histograms
            axes[1, 0].plot(0, 0, 'bo', label='Point')
            axes[1, 0].annotate(f'({0}, {0})', xy=(0, 0), xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')
            axes[1, 0].plot(3*voxel_side, 3*voxel_side, 'bo', label='Point')
            axes[1, 0].annotate(f'({round(3*voxel_side,1)}, {round(3*voxel_side,1)})', xy=(3*voxel_side, 3*voxel_side), xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')
            axes[1, 0].plot(1.5*voxel_side, 1.5*voxel_side, 'bo', label='Point')
            axes[1, 0].annotate(f'({round(1.5*voxel_side,1)}, {round(1.5*voxel_side,1)})', xy=(1.5*voxel_side, 1.5*voxel_side), xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')
            axes[1, 0].grid(True)
            axes[1, 0].set_facecolor('whitesmoke')
            axes[1, 0].set_title('VEGF concentration', fontsize=font)

            axes[1, 1].set_xlim(-size, size)
            axes[1, 1].set_ylim(-size, size)
            world.show_tumor_slice(axes[1, 1], fig, 'number_of_necrotic_cells', levels= np.linspace(1, 1001, 11), cmap='viridis', extend = 'neither')
            #plot the three voxels that are used for o2 histograms
            axes[1, 1].plot(0, 0, 'bo', label='Point')
            axes[1, 1].annotate(f'({0}, {0})', xy=(0, 0), xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')
            axes[1, 1].plot(3*voxel_side, 3*voxel_side, 'bo', label='Point')
            axes[1, 1].annotate(f'({round(3*voxel_side,1)}, {round(3*voxel_side,1)})', xy=(3*voxel_side, 3*voxel_side), xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')
            axes[1, 1].plot(1.5*voxel_side, 1.5*voxel_side, 'bo', label='Point')
            axes[1, 1].annotate(f'({round(1.5*voxel_side,1)}, {round(1.5*voxel_side,1)})', xy=(1.5*voxel_side, 1.5*voxel_side), xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')
            axes[1, 1].grid(True)
            axes[1, 1].set_facecolor('whitesmoke')
            axes[1, 1].set_title('Number of Necrotic Cells', fontsize=font)

            plt.tight_layout()
            plt.savefig('Plots/CurrentPlotting/t' + str(t) + '_AllPlots.png', dpi=100)
            if self.config.running_on_cluster:
                plt.close()
            else:
                plt.show()

        if self.config.show_o2_vitality_histograms:
            print('Showing histograms')
            voxel_side = 2*self.config.half_length_world / self.config.voxel_per_side

            voxels_positions = [[0,0,0], [1.5*voxel_side, 1.5*voxel_side, 1.5*voxel_side], [3*voxel_side, 3*voxel_side, 3*voxel_side]]
            fig, axes = plt.subplots(nrows=2, ncols=len(voxels_positions), figsize=(20, 10), dpi=100)
            fig.suptitle('Visualization at time t = ' + str(t) + ' hours', fontsize=16)
            count_under_threshold=0
            for i in range(len(voxels_positions)):
                #show histograms for the three voxels
                axes[0, i].set_title('Voxel ' + str(i))
                axes[0, i].set_xlabel('Oxygen')
                axes[0, i].set_ylabel('Number of cells')
                voxel = world.find_voxel(voxels_positions[i])
                voxel.cycling_time_and_age_histogram(axes[0, i], fig)
                axes[1, i].set_xlabel('Vitality')
                axes[1, i].set_ylabel('Number of cells')
                voxel.vitality_histogram(axes[1, i], fig)
                axes[1, i].axvline(x=self.config.vitality_cycling_threshold, color='red', linestyle='--', linewidth=1)

                vitality = []
                for cell in voxel.list_of_cells:
                        vitality.append(cell.vitality())
                count_under_threshold += sum(v < self.config.vitality_cycling_threshold for v in vitality)
                
            print(f"Number of cells under vitality threshold < {self.config.vitality_cycling_threshold} : {count_under_threshold}")
                #axes[1, i].text(self.config.vitality_cycling_threshold + 0.5, axes[0, i].get_ylim()[1]*0.9, f'Threshold quiescent/cycling = {self.config.vitality_cycling_threshold}', color='red', fontsize=8)


            plt.tight_layout()
            plt.savefig('Plots/CurrentPlotting/t' + str(t) + '_O2_Vitality.png', dpi=100)
            if self.config.running_on_cluster:
                plt.close()
            else:
                plt.show()

        if self.config.show_cell_damage:
            print('Showing cell damage')
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=100)
            fig.suptitle('Visualization at time t = ' + str(t) + ' hours', fontsize=16)
            world.show_tumor_slice(axes, fig, 'average_cell_damage', levels= np.linspace(0, 1, 10), cmap='cool', extend = 'neither')
            if self.config.running_on_cluster:
                plt.close()
            else:
                plt.show()

        end = current_time()
        print('Time elapsed for showing graphs: ' + str(end - start) + ' seconds')

        if self.config.show_CT:
            #Compute HU value for each voxel
            world_HU=np.zeros((world.config.voxel_per_side,world.config.voxel_per_side,world.config.voxel_per_side))
            for voxel in world.voxel_list:
                HU=voxel.biological_to_HU(world.config.vitality_cycling_threshold)
                #if HU>10:
                    #print('HU value of voxel',voxel.voxel_number,'is',HU)
                i,j,k=world.index_to_ijk(voxel.voxel_number)
                world_HU[i,j,k]=HU

            #Apply scanner PSF 
            sigma_xy=0.5   #resolution in xy-plan in mm
            sigma_z=0.5    #axial resolution in mm
            
            voxel_length = 2 * world.half_length / world.number_of_voxels
            sigma_xy_pix=sigma_xy/voxel_length
            sigma_z_pix=sigma_z/voxel_length

            #Gaussian convolution for PSF
            #world_HU=gaussian_filter(world_HU,sigma=[sigma_z_pix,sigma_xy_pix,sigma_xy_pix])

            #Adding noise based on a gaussian distribution (could be poisson with the number of photons detected)
            #shape=world.config.voxel_per_side,world.config.voxel_per_side,world.config.voxel_per_side
            #world_HU+=np.random.normal(0,5,shape)   #adapt the sd

            ### Show middle slices
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            plt.subplots_adjust(bottom=0.25)
            fig.suptitle(
                f'Interactive Visualization of CT scan and Signal Distribution\n'
                f'Time = {self.time} h',
                fontsize=16
            )
        
            #Initial position
            init_z = world.config.voxel_per_side // 2
            init_y = world.config.voxel_per_side // 2
            init_x = world.config.voxel_per_side // 2
        
            #Initial show
            im_axial = axes[0, 0].imshow(world_HU[init_z, :, :], cmap=plt.cm.gray, vmin=0, vmax=140)
            axes[0, 0].set_title(f'Axial (z={init_z})')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')
        
            im_sagittal = axes[0, 1].imshow(world_HU[:, :, init_x], cmap=plt.cm.gray, vmin=0, vmax=140)
            axes[0, 1].set_title(f'Sagittal (x={init_x})')
            axes[0, 1].set_xlabel('Y')
            axes[0, 1].set_ylabel('Z')
        
            im_coronal = axes[1, 0].imshow(world_HU[:, init_y, :], cmap=plt.cm.gray, vmin=0, vmax=140)
            axes[1, 0].set_title(f'Coronal (y={init_y})')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Z')
        
            ### Show histograms HU
            axes[1, 1].hist(world_HU.flatten(), bins=50, alpha=0.7, color='skyblue')
            axes[1, 1].set_title('HU values distribution')
            axes[1, 1].set_xlabel('HU')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
            # Colorbar
            plt.colorbar(im_axial, ax=axes[0, 0], label='HU')
        
            # Cursors
            ax_z = plt.axes([0.1, 0.15, 0.2, 0.03])
            ax_y = plt.axes([0.4, 0.15, 0.2, 0.03])
            ax_x = plt.axes([0.7, 0.15, 0.2, 0.03])
        
            slider_z = Slider(ax_z, 'Z', 0, world.config.voxel_per_side-1, valinit=init_z, valstep=1)
            slider_y = Slider(ax_y, 'Y', 0, world.config.voxel_per_side-1, valinit=init_y, valstep=1)
            slider_x = Slider(ax_x, 'X', 0, world.config.voxel_per_side-1, valinit=init_x, valstep=1)
        
            # Cursors for "windowing process"
            ax_vmin = plt.axes([0.1, 0.10, 0.2, 0.03])
            ax_vmax = plt.axes([0.4, 0.10, 0.2, 0.03])
        
            slider_vmin = Slider(ax_vmin, 'Min HU', -30, 70, valinit=0, valstep=5)
            slider_vmax = Slider(ax_vmax, 'Max HU', 0, 200, valinit=60, valstep=5)
        
            def update(val):
                z = int(slider_z.val)
                y = int(slider_y.val)
                x = int(slider_x.val)
                vmin = slider_vmin.val
                vmax = slider_vmax.val
            
                # Update images
                im_axial.set_array(world_HU[z, :, :])
                im_axial.set_clim(vmin, vmax)
                axes[0, 0].set_title(f'Axial (z={z}) - HU: {world_HU[z, y, x]:.1f}')
            
                im_sagittal.set_array(world_HU[:, :, x])
                im_sagittal.set_clim(vmin, vmax)
                axes[0, 1].set_title(f'Sagittal (x={x})')
            
                im_coronal.set_array(world_HU[:, y, :])
                im_coronal.set_clim(vmin, vmax)
                axes[1, 0].set_title(f'Coronal (y={y})')
            
                # Crossed cursors 
                axes[0, 0].clear()
                axes[0, 0].imshow(world_HU[z, :, :], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
                axes[0, 0].axhline(y=y, color='red', linewidth=1, alpha=0.7)
                axes[0, 0].axvline(x=x, color='red', linewidth=1, alpha=0.7)
                axes[0, 0].set_title(f'Axial (z={z}) - HU: {world_HU[z, y, x]:.1f}')
            
                fig.canvas.draw_idle()
        
            # Event connexion
            slider_z.on_changed(update)
            slider_y.on_changed(update)
            slider_x.on_changed(update)
            slider_vmin.on_changed(update)
            slider_vmax.on_changed(update)
        
            plt.show()

            if self.config.export_dicom_stack:
                time_folder = os.path.join("amber/Plots", f"t{self.time:04d}")
                os.makedirs(time_folder, exist_ok=True)

                for z in range(world.config.voxel_per_side):
                    slice_data = ((world_HU[z, :, :] + 1024) * 1).astype(np.uint16)  #conversion in uint16 for DICOM

                    # Metadata mandatory
                    file_meta = FileMetaDataset()
                    file_meta.MediaStorageSOPClassUID = generate_uid()
                    file_meta.MediaStorageSOPInstanceUID = generate_uid()
                    file_meta.ImplementationClassUID = generate_uid()
                    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

                    # Head of dataset
                    filename = os.path.join(time_folder, f't{self.time:04d}_slice_{z:03d}.dcm')
                    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

                    # DICOM field
                    ds.PatientName = "AMBER_Simulation"
                    ds.PatientID = "123456"
                    ds.SliceLocation = z * voxel_length
                    ds.StudyInstanceUID = generate_uid()
                    ds.SeriesInstanceUID = generate_uid()
                    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
                    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

                    dt = datetime.datetime.now()
                    ds.StudyDate = dt.strftime('%Y%m%d')
                    ds.StudyTime = dt.strftime('%H%M%S')

                    # Dimensions and pixels
                    ds.Rows, ds.Columns = slice_data.shape
                    ds.PixelRepresentation = 0
                    ds.SamplesPerPixel = 1
                    ds.PhotometricInterpretation = "MONOCHROME2"
                    ds.BitsStored = 16
                    ds.BitsAllocated = 16
                    ds.HighBit = 15
                    ds.RescaleSlope = 1
                    ds.RescaleIntercept = -1024
                    ds.PixelData = slice_data.tobytes()

                    # Encoding
                    ds.is_little_endian = True
                    ds.is_implicit_VR = False

                    # Backup
                    ds.save_as(filename)

                print(f"DICOM exported in {time_folder}")
    
        def volume_rendering_3d(self, threshold=50, opacity=0.3):

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
            # 3D mesh
            z, y, x = np.mgrid[0:world.number_of_voxels, 0:world.number_of_voxels, 0:world.number_of_voxels]
        
            # Mask for voxel higher than threshold
            mask = world_HU > threshold
        
            # Scatter plot 3D
            scatter = ax.scatter(x[mask], y[mask], z[mask], 
                            c=world_HU[mask], 
                            cmap=plt.cm.gray, 
                            alpha=opacity,
                            s=1)
        
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(f'3D mesh (HU > {threshold})')
        
            plt.colorbar(scatter, label='HU')
            plt.show()

        if self.config.show_MRI:

            world_MRI = np.zeros((world.config.voxel_per_side,) * 3)

            for voxel in world.voxel_list:
                MRI_signal = voxel.MRI_voxel_intensity(
                    vitality_threshold=world.config.vitality_cycling_threshold,
                    MRI_sequence=world.config.MRI_sequence,
                    T1_base=world.config.T1_base,
                    T2_base=world.config.T2_base,
                    PD_base=world.config.PD_base,
                    TE=world.config.MRI_TE,
                    TR=world.config.MRI_TR,
                    TI=world.config.MRI_TI
                )
                i, j, k = world.index_to_ijk(voxel.voxel_number)
                world_MRI[i, j, k] = MRI_signal

            #world_MRI -= world_MRI.min()
            #if world_MRI.max() != 0:
            #    world_MRI /= world_MRI.max()


            # PSF
            sigma_xy = 0.5   # resolution mm
            sigma_z = 0.5    # resolution mm
            voxel_length = 2 * world.half_length / world.number_of_voxels
            sigma_xy_pix = sigma_xy / voxel_length
            sigma_z_pix = sigma_z / voxel_length

            world_MRI = gaussian_filter(world_MRI, sigma=[sigma_z_pix, sigma_xy_pix, sigma_xy_pix])

            # Adding noise
            # world_MRI += np.random.normal(0, 0.01, world_MRI.shape)

            # Interactive visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            plt.subplots_adjust(bottom=0.25)
            fig.suptitle(
                f'Interactive Visualization of MRI {world.config.MRI_sequence} and Signal Distribution\n'
                f'TE = {world.config.MRI_TE} ms, TR = {world.config.MRI_TR} ms, time = {self.time} h',
                fontsize=16
            )

            vmin = np.min(world_MRI)
            vmax = np.max(world_MRI)


            init_z = world.config.voxel_per_side // 2
            init_y = world.config.voxel_per_side // 2
            init_x = world.config.voxel_per_side // 2

            im_axial = axes[0, 0].imshow(world_MRI[init_z, :, :], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f'Axial (z={init_z})')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')

            im_sagittal = axes[0, 1].imshow(world_MRI[:, :, init_x], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f'Sagittal (x={init_x})')
            axes[0, 1].set_xlabel('Y')
            axes[0, 1].set_ylabel('Z')

            im_coronal = axes[1, 0].imshow(world_MRI[:, init_y, :], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            axes[1, 0].set_title(f'Coronal (y={init_y})')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Z')

            axes[1, 1].hist(world_MRI.flatten(), bins=50, alpha=0.7, color='gray')
            axes[1, 1].set_title('MRI Signal distribution')
            axes[1, 1].set_xlabel('Signal intensity')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)

            plt.colorbar(im_axial, ax=axes[0, 0], label='MRI Intensity')

            ax_z = plt.axes([0.1, 0.15, 0.25, 0.03])
            ax_y = plt.axes([0.4, 0.15, 0.25, 0.03])
            ax_x = plt.axes([0.7, 0.15, 0.25, 0.03])

            slider_z = Slider(ax_z, 'Z', 0, world.config.voxel_per_side - 1, valinit=init_z, valstep=1)
            slider_y = Slider(ax_y, 'Y', 0, world.config.voxel_per_side - 1, valinit=init_y, valstep=1)
            slider_x = Slider(ax_x, 'X', 0, world.config.voxel_per_side - 1, valinit=init_x, valstep=1)

            def update(val):
                z = int(slider_z.val)
                y = int(slider_y.val)
                x = int(slider_x.val)

                im_axial.set_array(world_MRI[z, :, :])
                axes[0, 0].set_title(f'Axial (z={z}) - Intensity: {world_MRI[z, y, x]:.2f}')

                im_sagittal.set_array(world_MRI[:, :, x])
                axes[0, 1].set_title(f'Sagittal (x={x})')

                im_coronal.set_array(world_MRI[:, y, :])
                axes[1, 0].set_title(f'Coronal (y={y})')

                fig.canvas.draw_idle()

            slider_z.on_changed(update)
            slider_y.on_changed(update)
            slider_x.on_changed(update)

            plt.show()

            if self.config.export_dicom_stack_MRI:
                time_folder = os.path.join("amber/Plots", f"MRI_t{self.time:04d}")
                os.makedirs(time_folder, exist_ok=True)

                for z in range(world.config.voxel_per_side):
                    # Normalize signal to 8-bit grayscale (0–255)
                    slice_data = (world_MRI[z, :, :] * 255).astype(np.uint8)

                    # DICOM mandatory file metadata
                    file_meta = FileMetaDataset()
                    file_meta.MediaStorageSOPClassUID = generate_uid()
                    file_meta.MediaStorageSOPInstanceUID = generate_uid()
                    file_meta.ImplementationClassUID = generate_uid()
                    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

                    # Full path for slice DICOM file
                    filename = os.path.join(time_folder, f'MRI_t{self.time:04d}_slice_{z:03d}.dcm')

                    # Create the dataset
                    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

                    # Basic patient/study info
                    ds.PatientName = "AMBER_MRI_Simulation"
                    ds.PatientID = "123456"
                    ds.Modality = "MR"
                    ds.SeriesDescription = f"Synthetic MRI ({world.config.MRI_sequence})"
                    ds.ProtocolName = world.config.MRI_sequence  # Standard field to specify sequence type
                    ds.StudyInstanceUID = generate_uid()
                    ds.SeriesInstanceUID = generate_uid()
                    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
                    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

                    # Position and orientation
                    ds.SliceLocation = z * voxel_length
                    ds.ImagePositionPatient = [0, 0, z * voxel_length]
                    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # axial orientation
                    ds.PixelSpacing = [voxel_length, voxel_length]
                    ds.SliceThickness = voxel_length

                    # Time/date
                    dt = datetime.datetime.now()
                    ds.StudyDate = dt.strftime('%Y%m%d')
                    ds.StudyTime = dt.strftime('%H%M%S')

                    # Image properties
                    ds.Rows, ds.Columns = slice_data.shape
                    ds.PixelRepresentation = 0  # unsigned integers
                    ds.SamplesPerPixel = 1
                    ds.PhotometricInterpretation = "MONOCHROME2"
                    ds.BitsStored = 8
                    ds.BitsAllocated = 8
                    ds.HighBit = 7

                    # Pixel data
                    ds.PixelData = slice_data.tobytes()

                    # Encoding details
                    ds.is_little_endian = True
                    ds.is_implicit_VR = False

                    # Save DICOM file
                    ds.save_as(filename)

                print(f"MRI DICOM exported in {time_folder}")

        if self.config.show_MRI_DWI_nec:

            world_DWI = np.zeros((world.config.voxel_per_side,) * 3)

            for voxel in world.voxel_list:
                DWI_signal = voxel.DWI_MRI_intensity(bvalue=world.config.bvalue, 
                                                     bvecs=world.config.bvecs, 
                                                     TD=world.config.TD, 
                                                     pulsew=4, 
                                                     Din=1, 
                                                     Dex=world.config.Dex, 
                                                     kappa=0.03)
                i, j, k = world.index_to_ijk(voxel.voxel_number)
                world_DWI[i, j, k] = DWI_signal

            # PSF
            sigma_xy = 0.5   # resolution mm
            sigma_z = 0.5    # resolution mm
            voxel_length = 2 * world.half_length / world.number_of_voxels
            sigma_xy_pix = sigma_xy / voxel_length
            sigma_z_pix = sigma_z / voxel_length

            world_DWI = gaussian_filter(world_DWI, sigma=[sigma_z_pix, sigma_xy_pix, sigma_xy_pix])

            # Adding noise
            # world_MRI += np.random.normal(0, 0.01, world_DWI.shape)

            # Interactive visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            plt.subplots_adjust(bottom=0.25)
            fig.suptitle(
                f'Interactive Visualization of DWI and Signal Distribution\n'
                f'b = {world.config.bvalue} ms/um², TD = {world.config.TD} ms, Dex = {world.config.Dex}, time = {self.time} h',
                fontsize=16
            )


            #vmin = np.min(world_DWI)
            #vmax = np.max(world_DWI)

            vmin=0
            vmax=1


            init_z = world.config.voxel_per_side // 2
            init_y = world.config.voxel_per_side // 2
            init_x = world.config.voxel_per_side // 2

            im_axial = axes[0, 0].imshow(world_DWI[init_z, :, :], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f'Axial (z={init_z})')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')

            im_sagittal = axes[0, 1].imshow(world_DWI[:, :, init_x], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f'Sagittal (x={init_x})')
            axes[0, 1].set_xlabel('Y')
            axes[0, 1].set_ylabel('Z')

            im_coronal = axes[1, 0].imshow(world_DWI[:, init_y, :], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            axes[1, 0].set_title(f'Coronal (y={init_y})')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Z')

            # Flatten data
            data = world_DWI.flatten()

            # Histogram
            counts, bin_edges = np.histogram(data, bins=50)

            # Get rid of bin with frequence zero
            nonzero = counts > 0
            counts = counts[nonzero]

            # Bins center and width
            bin_lefts = bin_edges[:-1]
            bin_rights = bin_edges[1:]
            bin_centers = (bin_lefts + bin_rights) / 2
            bin_centers = bin_centers[nonzero]
            bin_widths = (bin_rights - bin_lefts)[nonzero]

            # Plot
            axes[1, 1].bar(bin_centers, counts, width=bin_widths, color='gray', alpha=0.7)
            axes[1, 1].set_title('Diffusion MRI Signal distribution (log scale)')
            axes[1, 1].set_xlabel('Signal intensity')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, which='both', axis='y', alpha=0.3)

            plt.colorbar(im_axial, ax=axes[0, 0], label='Diffusion MRI Intensity')

            ax_z = plt.axes([0.1, 0.15, 0.25, 0.03])
            ax_y = plt.axes([0.4, 0.15, 0.25, 0.03])
            ax_x = plt.axes([0.7, 0.15, 0.25, 0.03])

            slider_z = Slider(ax_z, 'Z', 0, world.config.voxel_per_side - 1, valinit=init_z, valstep=1)
            slider_y = Slider(ax_y, 'Y', 0, world.config.voxel_per_side - 1, valinit=init_y, valstep=1)
            slider_x = Slider(ax_x, 'X', 0, world.config.voxel_per_side - 1, valinit=init_x, valstep=1)

            def update(val):
                z = int(slider_z.val)
                y = int(slider_y.val)
                x = int(slider_x.val)

                im_axial.set_array(world_DWI[z, :, :])
                axes[0, 0].set_title(f'Axial (z={z}) - Intensity: {world_DWI[z, y, x]:.2f}')

                im_sagittal.set_array(world_DWI[:, :, x])
                axes[0, 1].set_title(f'Sagittal (x={x})')

                im_coronal.set_array(world_DWI[:, y, :])
                axes[1, 0].set_title(f'Coronal (y={y})')

                fig.canvas.draw_idle()

            slider_z.on_changed(update)
            slider_y.on_changed(update)
            slider_x.on_changed(update)

            plt.show()

            if self.config.export_dicom_stack_DWI:
                time_folder = os.path.join("amber/Plots", f"DWI_t{self.time:04d}")
                os.makedirs(time_folder, exist_ok=True)

                for z in range(world.config.voxel_per_side):
                    # Normalize signal to 8-bit grayscale (0–255)
                    slice_data = (world_DWI[z, :, :] * 255).astype(np.uint8)

                    # DICOM mandatory file metadata
                    file_meta = FileMetaDataset()
                    file_meta.MediaStorageSOPClassUID = generate_uid()
                    file_meta.MediaStorageSOPInstanceUID = generate_uid()
                    file_meta.ImplementationClassUID = generate_uid()
                    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

                    # Full path for slice DICOM file
                    filename = os.path.join(time_folder, f'DWI_t{self.time:04d}_slice_{z:03d}.dcm')

                    # Create the dataset
                    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

                    # Basic patient/study info
                    ds.PatientName = "AMBER_Diffusion_MRI_Simulation"
                    ds.PatientID = "123456"
                    ds.Modality = "MR"
                    ds.SeriesDescription = f"Synthetic MRI (DWI)"
                    ds.ProtocolName = "DWI"
                    ds.StudyInstanceUID = generate_uid()
                    ds.SeriesInstanceUID = generate_uid()
                    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
                    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

                    # Position and orientation
                    ds.SliceLocation = z * voxel_length
                    ds.ImagePositionPatient = [0, 0, z * voxel_length]
                    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # axial orientation
                    ds.PixelSpacing = [voxel_length, voxel_length]
                    ds.SliceThickness = voxel_length

                    # Time/date
                    dt = datetime.datetime.now()
                    ds.StudyDate = dt.strftime('%Y%m%d')
                    ds.StudyTime = dt.strftime('%H%M%S')

                    # Image properties
                    ds.Rows, ds.Columns = slice_data.shape
                    ds.PixelRepresentation = 0  # unsigned integers
                    ds.SamplesPerPixel = 1
                    ds.PhotometricInterpretation = "MONOCHROME2"
                    ds.BitsStored = 8
                    ds.BitsAllocated = 8
                    ds.HighBit = 7

                    # Pixel data
                    ds.PixelData = slice_data.tobytes()

                    # Encoding details
                    ds.is_little_endian = True
                    ds.is_implicit_VR = False

                    # Save DICOM file
                    ds.save_as(filename)

                print(f"Diffusion MRI DICOM exported in {time_folder}")
        
        if self.config.show_MRI_DWI_nec:

            world_DWI_nec = np.zeros((world.config.voxel_per_side,) * 3)

            for voxel in world.voxel_list:
                DWI_signal_nec = voxel.DWI_MRI_intensity_nec(bvalue=world.config.bvalue, 
                                                    bvecs=world.config.bvecs, 
                                                    TD=world.config.TD, 
                                                    pulsew=4, 
                                                    Din=1, 
                                                    Dex=world.config.Dex, 
                                                    kappa_dead=world.config.kappa_dead)
                i, j, k = world.index_to_ijk(voxel.voxel_number)
                world_DWI_nec[i, j, k] = DWI_signal_nec

            # PSF
            sigma_xy = 0.5   # resolution mm
            sigma_z = 0.5    # resolution mm
            voxel_length = 2 * world.half_length / world.number_of_voxels
            sigma_xy_pix = sigma_xy / voxel_length
            sigma_z_pix = sigma_z / voxel_length

            world_DWI_nec = gaussian_filter(world_DWI_nec, sigma=[sigma_z_pix, sigma_xy_pix, sigma_xy_pix])

            # Adding noise
            # world_MRI += np.random.normal(0, 0.01, world_DWI.shape)

            # Interactive visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            plt.subplots_adjust(bottom=0.25)
            fig.suptitle(
                f'Interactive Visualization of DWI and Signal Distribution with necrosis\n'
                f'b = {world.config.bvalue} ms/um², TD = {world.config.TD} ms, Dex = {world.config.Dex}, kappa_dead = {world.config.kappa_dead}, time = {self.time} h',
                fontsize=16
            )


            #vmin = np.min(world_DWI)
            #vmax = np.max(world_DWI)

            vmin=0
            vmax=1


            init_z = world.config.voxel_per_side // 2
            init_y = world.config.voxel_per_side // 2
            init_x = world.config.voxel_per_side // 2

            im_axial = axes[0, 0].imshow(world_DWI_nec[init_z, :, :], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f'Axial (z={init_z})')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')

            im_sagittal = axes[0, 1].imshow(world_DWI_nec[:, :, init_x], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f'Sagittal (x={init_x})')
            axes[0, 1].set_xlabel('Y')
            axes[0, 1].set_ylabel('Z')

            im_coronal = axes[1, 0].imshow(world_DWI_nec[:, init_y, :], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            axes[1, 0].set_title(f'Coronal (y={init_y})')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Z')

            # Flatten data
            data = world_DWI_nec.flatten()

            # Histogram
            counts, bin_edges = np.histogram(data, bins=50)

            # Get rid of bin with frequence zero
            nonzero = counts > 0
            counts = counts[nonzero]

            # Bins center and width
            bin_lefts = bin_edges[:-1]
            bin_rights = bin_edges[1:]
            bin_centers = (bin_lefts + bin_rights) / 2
            bin_centers = bin_centers[nonzero]
            bin_widths = (bin_rights - bin_lefts)[nonzero]

            # Plot
            axes[1, 1].bar(bin_centers, counts, width=bin_widths, color='gray', alpha=0.7)
            axes[1, 1].set_title('Diffusion MRI with necrosis Signal distribution (log scale)')
            axes[1, 1].set_xlabel('Signal intensity')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, which='both', axis='y', alpha=0.3)

            plt.colorbar(im_axial, ax=axes[0, 0], label='Diffusion MRI with necrosis Intensity')

            ax_z = plt.axes([0.1, 0.15, 0.25, 0.03])
            ax_y = plt.axes([0.4, 0.15, 0.25, 0.03])
            ax_x = plt.axes([0.7, 0.15, 0.25, 0.03])

            slider_z = Slider(ax_z, 'Z', 0, world.config.voxel_per_side - 1, valinit=init_z, valstep=1)
            slider_y = Slider(ax_y, 'Y', 0, world.config.voxel_per_side - 1, valinit=init_y, valstep=1)
            slider_x = Slider(ax_x, 'X', 0, world.config.voxel_per_side - 1, valinit=init_x, valstep=1)

            def update(val):
                z = int(slider_z.val)
                y = int(slider_y.val)
                x = int(slider_x.val)

                im_axial.set_array(world_DWI_nec[z, :, :])
                axes[0, 0].set_title(f'Axial (z={z}) - Intensity: {world_DWI[z, y, x]:.2f}')

                im_sagittal.set_array(world_DWI_nec[:, :, x])
                axes[0, 1].set_title(f'Sagittal (x={x})')

                im_coronal.set_array(world_DWI_nec[:, y, :])
                axes[1, 0].set_title(f'Coronal (y={y})')

                fig.canvas.draw_idle()

            slider_z.on_changed(update)
            slider_y.on_changed(update)
            slider_x.on_changed(update)

            plt.show()

            if self.config.export_dicom_stack_DWI_nec:
                time_folder = os.path.join("amber/Plots", f"DWI_nec_t{self.time:04d}")
                os.makedirs(time_folder, exist_ok=True)

                for z in range(world.config.voxel_per_side):
                    # Normalize signal to 8-bit grayscale (0–255)
                    slice_data = (world_DWI_nec[z, :, :] * 255).astype(np.uint8)

                    # DICOM mandatory file metadata
                    file_meta = FileMetaDataset()
                    file_meta.MediaStorageSOPClassUID = generate_uid()
                    file_meta.MediaStorageSOPInstanceUID = generate_uid()
                    file_meta.ImplementationClassUID = generate_uid()
                    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

                    # Full path for slice DICOM file
                    filename = os.path.join(time_folder, f'DWI_nec_t{self.time:04d}_slice_{z:03d}.dcm')

                    # Create the dataset
                    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

                    # Basic patient/study info
                    ds.PatientName = "AMBER_Diffusion_MRI_Simulation_with_necrosis"
                    ds.PatientID = "123456"
                    ds.Modality = "MR"
                    ds.SeriesDescription = f"Synthetic MRI (DWI) with necrosis"
                    ds.ProtocolName = "DWI with necrosis"
                    ds.StudyInstanceUID = generate_uid()
                    ds.SeriesInstanceUID = generate_uid()
                    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
                    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

                    # Position and orientation
                    ds.SliceLocation = z * voxel_length
                    ds.ImagePositionPatient = [0, 0, z * voxel_length]
                    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # axial orientation
                    ds.PixelSpacing = [voxel_length, voxel_length]
                    ds.SliceThickness = voxel_length

                    # Time/date
                    dt = datetime.datetime.now()
                    ds.StudyDate = dt.strftime('%Y%m%d')
                    ds.StudyTime = dt.strftime('%H%M%S')

                    # Image properties
                    ds.Rows, ds.Columns = slice_data.shape
                    ds.PixelRepresentation = 0  # unsigned integers
                    ds.SamplesPerPixel = 1
                    ds.PhotometricInterpretation = "MONOCHROME2"
                    ds.BitsStored = 8
                    ds.BitsAllocated = 8
                    ds.HighBit = 7

                    # Pixel data
                    ds.PixelData = slice_data.tobytes()

                    # Encoding details
                    ds.is_little_endian = True
                    ds.is_implicit_VR = False

                    # Save DICOM file
                    ds.save_as(filename)

                print(f"Diffusion MRI with necrosis DICOM exported in {time_folder}")

    def run(self, world: World, video=False): #run the simulation! (the main function)

        print(f'Running simulation for {self.finish_time} hours with dt={self.dt}')
        process_local = [process for process in self.list_of_process if not process.is_global] #list of local processes
        process_global = [process for process in self.list_of_process if process.is_global] #list of global processes
        print('Number of local processes:', len(process_local))
        print('Number of global processes:', len(process_global))


        #create an array of execution times for each process (local and global)
        execution_times = [[] for _ in range(len(self.list_of_process))] #list of execution times for each process (local and global)
        sums = []

        # if not self.config.skip_weekends:
        #     irradiations_times = [self.config.first_irradiation_time + i * self.config.time_between_fractions for i in
        #                           range(self.config.number_fractions)] #list of irradiation times
        #read the fractionation schedule from a file
        irradiations_times = []
        with open(str(self.config.fractionation_schedule), 'r') as f:
            for line in f:
                irradiations_times.append(float(line))

        print('Irradiation times: ' + str(irradiations_times))
        can_irradiate = False
        if not self.config.irradiation_cell_based:
            can_irradiate = True
            for i in range(len(irradiations_times)):
                irradiations_times[i] = irradiations_times[i] + self.config.first_irradiation_time

        number_of_fractions = len(irradiations_times) #number of fractions
        applied_fractions = 0 #number of applied fractions

        number_cycling_cells = []; number_quiescent_cells = []; number_necrotic_cells = [];
        tumor_size = []; tumor_size_free = []; times = []; number_tumor_cells = []; number_vessels = [];
        center_of_mass = []

        while self.time < self.finish_time: #while the simulation is running
            print(f'\033[1;31;47mTime: {self.time} hours / {self.finish_time} hours\033[0m')
            #show the simulation
            if video:
                self.show(world, self.time) #show the simulation

            #do irradiation if needed
            if can_irradiate and applied_fractions < number_of_fractions and self.time >= irradiations_times[applied_fractions]:
                irrad = Irradiation(self.config, 'irrad', self.dt, self.config.topas_file,
                                    self.config.irradiation_intensity, world)
                irrad(world)
                applied_fractions += 1
                print('Irradiation at time',current_time())

            print('Currently running local processes:')
            start = current_time()
            for voxel in world.voxel_list:
                for process in process_local:
                    process(voxel) #run local processes on each voxel
            end = current_time()
            print('Time for locals processes:', end - start)
            for process in process_global: #run global processes on the whole world
                print('Currently running global process:', process.name)
                start = current_time()
                process(world)
                end = current_time()
                print('Time for process', process.name, ':', end - start)

            cycling_cells = 0
            quiescent_cells = 0
            necrotic_cells = 0
            start = current_time()
            for voxel in world.voxel_list: #count the number of cells in each state
                necrotic_cells += voxel.number_of_necrotic_cells()
                for cell in voxel.list_of_cells:
                    if cell.type == 'TumorCell':
                        if cell.vitality() > self.config.vitality_cycling_threshold:
                            cycling_cells += 1
                        else:
                            quiescent_cells += 1
            end = current_time()
            print('Time for counting cells:', end - start)
            print('Number of cycling cells:', cycling_cells)
            print('Number of quiescent cells:', quiescent_cells)
            print('Number of necrotic cells:', necrotic_cells)
            print('Number of tumor cells:', cycling_cells + quiescent_cells + necrotic_cells)

            number_cycling_cells.append(cycling_cells)
            number_quiescent_cells.append(quiescent_cells)
            number_necrotic_cells.append(necrotic_cells)
            number_tumor_cells.append(cycling_cells + quiescent_cells + necrotic_cells)
            tumor_size_, tumor_size_free_ = world.measure_tumor_volume()
            tumor_size.append(tumor_size_)
            tumor_size_free.append(tumor_size_free_)
            number_vessels.append(len(world.vasculature.list_of_vessels))
            center_of_mass.append(world.center_of_mass)
            times.append(self.time)

            np.save('DataOutput/number_tumor_cells.npy', number_tumor_cells)
            np.save('DataOutput/number_necrotic_cells.npy', number_necrotic_cells)
            np.save('DataOutput/number_cycling_cells.npy', number_cycling_cells)
            np.save('DataOutput/number_quiescent_cells.npy', number_quiescent_cells)
            np.save('DataOutput/tumor_size.npy', tumor_size)
            np.save('DataOutput/tumor_size_free.npy', tumor_size_free)
            np.save('DataOutput/number_vessels.npy', number_vessels)
            np.save('DataOutput/center_of_mass.npy', center_of_mass)
            np.save('DataOutput/times.npy', times)

            if self.time % self.config.save_world_every == 0 and self.time != 0:
                world.save('t'+str(self.time)+str(self.config.world_file) + str(self.config.seed) + '.pkl')

            if self.config.show_cell_and_tumor_volume:
                self.show_cell_and_tumor_volume(number_tumor_cells, number_necrotic_cells, number_quiescent_cells, number_cycling_cells, tumor_size, tumor_size_free, number_vessels, times)


            if self.config.show_center_of_mass:
                self.center_of_mass(center_of_mass, times)

            sum = 0
            for i, process in enumerate(self.list_of_process):
                sum += process.time_spent_doing_process
                execution_times[i].append(process.time_spent_doing_process)
                process.time_spent_doing_process = 0
            sums.append(sum)

            if not can_irradiate and self.config.irradiation_cell_based and number_tumor_cells[-1] >= self.config.critical_n_cells:
                can_irradiate = True
                for i in range(len(irradiations_times)):
                    irradiations_times[i] = irradiations_times[i] + self.time

            #print('Time spent doing processes:', execution_times)
            plt.figure()
            for i, process in enumerate(self.list_of_process):
                plt.plot(times, execution_times[i], label = process.name)
            plt.plot(times, sums, label = 'Total')
            plt.legend()
            plt.xlabel('Time (hours)')
            plt.ylabel('Time spent doing process (s)')
            plt.yscale('log')
            plt.grid()
            plt.savefig('Plots/execution_times.png')
            plt.close()

            self.time += self.dt

        print('Simulation finished')

        if self.config.show_final:
            self.show(world, self.time)

        print('end of running')

        return

class Process: #abstract class, represents all the processes that can happen in the simulation

    time_spent_doing_process = 0
    def __init__(self, config, name, dt):
        self.name = name
        self.dt = dt
        self.is_global = False
        self.config = config

    def __call__(self, voxel):
        pass

    @classmethod #decorator to measure the time spent doing a process
    def timeit(cls, process_func):
        def wrapped_process(self, voxel):
            start_time = current_time()
            result = process_func(self, voxel)
            end_time = current_time()
            elapsed_time = end_time - start_time
            self.time_spent_doing_process += elapsed_time
            return result
        return wrapped_process


class CellDivision(Process): #cell division process, cells divide in a voxel if they have enough vitality
    def __init__(self, config, name, dt, cycling_threshold, pressure_threshold = np.inf):
        super().__init__(config, 'CellDivision', dt)
        self.dt = dt
        self.cycling_threshold = cycling_threshold
        self.pressure_threshold = pressure_threshold

    @Process.timeit
    def __call__(self, voxel):
        if len(voxel.list_of_cells) > 0:
            for cell in voxel.list_of_cells:
                if cell.time_spent_cycling >= cell.doubling_time:
                    time_diff = cell.time_spent_cycling - cell.doubling_time
                    leftover_time = self.dt - time_diff
                    new_cell = cell.duplicate() #create a new cell (start cycling at 0)
                    new_cell.time_spent_cycling = leftover_time #reset the time spent cycling
                    cell.doubling_time = cell.random_doubling_time() #sample a new doubling time for the old cell
                    cell.time_spent_cycling = leftover_time #reset the time spent cycling
                    voxel.add_cell(new_cell, self.config.max_occupancy) #add the new cell to the voxel
        return

class CellDeath(Process): #cell necrosis process, cells die in a voxel if they have too low vitality
    def __init__(self, config, name, dt, necrosis_threshold, necrosis_probability, apoptosis_threshold, apoptosis_probability,necrosis_removal_probability, apoptosis_removal_probability, necrosis_damage_coeff, apoptosis_damage_coeff):
        super().__init__(config, 'CellNecrosis', dt)
        self.necrosis_threshold = necrosis_threshold
        self.necrosis_probability = necrosis_probability
        self.apoptosis_threshold = apoptosis_threshold
        self.apoptosis_probability = apoptosis_probability
        self.necrosis_damage_coeff = necrosis_damage_coeff
        self.apoptosis_damage_coeff = apoptosis_damage_coeff
        if self.necrosis_threshold > self.apoptosis_threshold:
            raise ValueError('necrosis threshold must be smaller or equal to apoptosis threshold. you can set apoptosis probability to 0 if you want to avoid apoptosis.')
        self.necrosis_removal_probability = necrosis_removal_probability
        self.apoptosis_removal_probability = apoptosis_removal_probability

    @Process.timeit
    def __call__(self, voxel):
        dead_cell=0
        p_nec=0
        p_apo=0
        for cell in voxel.list_of_cells:
            sample = np.random.uniform(0, 1)
            #probability of necrosis and apoptosis. use math to get sampling every hour
            p_necro = (1 - ((1-cell.necrosis_probability(self.necrosis_probability,self.necrosis_threshold, self.necrosis_damage_coeff))**self.dt))
            p_apopt = (1 - ((1-cell.apoptosis_probability(self.apoptosis_probability, self.apoptosis_threshold, self.apoptosis_damage_coeff))**self.dt))
            if self.config.verbose: print('probability necro:', p_necro, 'probability apopto:', p_apopt)
            if p_necro>0:
                p_nec+=1
            if p_apopt>0:
                p_apo+=1
            p_tot = p_necro + p_apopt
            if p_tot > 1:
                p_necro = p_necro/p_tot; p_apopt = p_apopt/p_tot

            if sample < p_necro:
                #necrosis
                voxel.cell_becomes_necrotic(cell)
                dead_cell+=1
            elif sample < p_necro + p_apopt:
                #apoptosis
                voxel.cell_becomes_apoptotic(cell)
                dead_cell+=1
        #if (dead_cell>0):
            #print('Nb of dead cells in voxel',voxel.voxel_number,'is ',dead_cell)
        #if p_nec>0 or p_apo>0:
            #print('Nb of proba non zero is',p_nec,'for necrosis and ',p_apo,'for apoptosis')
        for dead in voxel.list_of_dead_cells: #remove dead cells with a certain probability
            if dead.necrotic: p = self.necrosis_removal_probability
            else: p = self.apoptosis_removal_probability
            proba = (1 - ((1-p)**self.dt))
            if random.random() < proba:
                voxel.remove_dead_cell(dead)

class CellAging(Process): #cell aging process, cells age in a voxel
    def __init__(self, config, name, dt, repair_per_hour):
        super().__init__(config,'CellAging', dt)
        self.repair_per_hour = repair_per_hour

    @Process.timeit
    def __call__(self, voxel):
        for cell in voxel.list_of_cells:

            if cell.is_cycling(self.config.vitality_cycling_threshold, 0.0): #cell aging towards duplication
                cell.time_spent_cycling += self.dt

            if cell.damage > 0: #cell being repaired
                cell.damage_repair(self.dt, self.repair_per_hour)

        pass

class CellInteraction(Process):

    def __init__(self, config, name, dt):
        super().__init__(config, 'CellInteraction', dt)
        self.dt = dt

    @Process.timeit
    def __call__(self, voxel): #cell interaction process, cells interact in a voxel
        matrix = voxel.compute_cell_interaction_matrix(self.dt)
        number_cells = len(voxel.list_of_cells)
        #find non zero elements of the sparse matrix
        idd = matrix.nonzero()
        for i in range(len(idd[0])):
            if np.random.random() < matrix[idd[0][i], idd[1][i]]:
                cell1 = voxel.list_of_cells[idd[0][i]]
                cell2 = voxel.list_of_cells[idd[1][i]]
                cell1.interact(cell2, self.dt) #do the interaction between the two cells

class CellMigration(Process): #cell migration process, cells migrate in the world
    def __init__(self, config, name, dt):
        super().__init__(config, 'CellMigration', dt)
        self.is_global = True #run on the whole world, after the other processes
    @Process.timeit
    def __call__(self, world: World):
        exchange_matrix = world.compute_exchange_matrix(self.dt) #compute the exchange matrix for the time step
        for voxel in world.voxel_list:
            voxel_num = voxel.voxel_number
            #print('(0,0,0)=',world.ijk_to_index(0,0,0))
            #print('(3,3,3)=',world.ijk_to_index(3,3,3))
            #print('(5,5,5)=',world.ijk_to_index(5,5,5))
            #if voxel_num==7812: print('7812=',world.index_to_ijk(voxel_num))
            if voxel_num % 10000 == 0: print('voxel number = ', voxel_num)

            list_of_neighbors = world.find_moor_neighbors(voxel)
            np.random.shuffle(list_of_neighbors) #shuffle the list to avoid bias
            for neighbor in list_of_neighbors:
                n_events = exchange_matrix[voxel_num, neighbor.voxel_number] #number of expected events in the time step
                n_moving_cells = np.random.poisson(n_events)
                #n_moving_cells = int(math.ceil(n_events))
                n_moving_cells = min(n_moving_cells, int(round(len(voxel.list_of_cells))))
                list_of_moving_cells = np.random.choice(voxel.list_of_cells, n_moving_cells, replace=False) #choose the cells to move randomly
                for cell in list_of_moving_cells: #move the cells
                    if neighbor.add_cell(cell, self.config.max_occupancy):
                        voxel.remove_cell(cell)

class UpdateCellOxygen(Process):
    def __init__(self, config, name, dt, voxel_half_length, file_prefix_alpha_beta_maps):
        super().__init__(config, 'UpdateState', dt)
        self.voxel_side = round(voxel_half_length*20,1) #um/100
        print('voxel size is',self.voxel_side)

        #read alpha and beta maps from csv file
        first_try = str(self.voxel_side)
        second_try = str(round(self.voxel_side))
        amber_dir = os.path.abspath(os.path.dirname(__file__))

        alpha_file_name_first = str(file_prefix_alpha_beta_maps) + '_alpha_dataframe' + str(first_try) + '.csv'
        beta_file_name_first = str(file_prefix_alpha_beta_maps) + '_beta_dataframe' + str(first_try) + '.csv'

        alpha_file_name_second = str(file_prefix_alpha_beta_maps) + '_alpha_dataframe' + str(second_try) + '.csv'
        beta_file_name_second = str(file_prefix_alpha_beta_maps) + '_beta_dataframe' + str(second_try) + '.csv'

        # Check if the first file exists, if not, use the second file name
        alpha_file_name = alpha_file_name_first if os.path.exists(
            os.path.join(amber_dir, alpha_file_name_first)) else alpha_file_name_second
        beta_file_name = beta_file_name_first if os.path.exists(
            os.path.join(amber_dir, beta_file_name_first)) else beta_file_name_second

        print("Oxygen Files are : ")
        print(first_try, second_try)
        print(alpha_file_name, beta_file_name)

        alpha_file_name = os.path.join(amber_dir, alpha_file_name)
        beta_file_name = os.path.join(amber_dir, beta_file_name)

        if not os.path.isfile(alpha_file_name) or not os.path.isfile(beta_file_name):
            print('voxel side', self.voxel_side)
            print('alpha file name is ', alpha_file_name)
            print('beta file name is ', beta_file_name)
            raise ValueError('alpha/beta file not found! It might be in the wrong directory or information for chosen voxel size is not stored. Check "BetaDistributionCalibration.py" to generate the file for the chosen voxel size.')

        alpha_dataframe = pd.read_csv(alpha_file_name, index_col=0)
        beta_dataframe = pd.read_csv(beta_file_name, index_col=0)

        pressure_column = alpha_dataframe.index.values
        n_column = alpha_dataframe.columns.values.astype(float)

        self.max_n = max(n_column)

        # Create a 2D grid of points (pressure, n)
        points = []
        values_alpha = []
        values_beta = []

        for p in pressure_column:
            for n in n_column:
                alpha_value = alpha_dataframe.at[p, str(n)]
                beta_value = beta_dataframe.at[p, str(n)]
                points.append([p, n])
                values_alpha.append(alpha_value)
                values_beta.append(beta_value)

        self.alpha_map = ScalarField2D(points, values_alpha, bounds_error=False, fill_value= None)
        self.beta_map = ScalarField2D(points, values_beta, bounds_error=False, fill_value= None)

        if self.config.show_alpha_beta_maps:
            # Plot the alpha and beta maps
            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(azim=45, elev=30)
            self.alpha_map.show_extra(fig, ax, [min(pressure_column), max(pressure_column)], [min(n_column), max(n_column)])
            ax.axes.set_xlabel('Cell density')
            ax.axes.set_ylabel('Number of capillaries')
            ax.title.set_text('Alpha map')
            plt.savefig('alpha_map.png', dpi=300)
            if self.config.running_on_cluster:
                plt.close()
            else:
                plt.show()

            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(azim=135, elev=30)
            self.beta_map.show_extra(fig, ax, [min(pressure_column), max(pressure_column)], [min(n_column), max(n_column)])
            ax.axes.set_xlabel('Cell density')
            ax.axes.set_ylabel('Number of capillaries')
            ax.title.set_text('Beta map')
            plt.savefig('beta_map.png', dpi=300)
            if self.config.running_on_cluster:
                plt.close()
            else:
                plt.show()
    @Process.timeit
    def __call__(self, voxel: Voxel):

        n_vessels = voxel.n_capillaries
        n_cells = voxel.number_of_alive_cells()
        pressure = voxel.pressure()
        if self.config.o2_off:
            o2_values = np.ones(n_cells)
        else:
            if n_vessels == 0:
                o2_values = np.zeros(n_cells) #if there are no vessels, all cells have 0 oxygen

            elif n_vessels > self.max_n:
                o2_values = np.ones(n_cells) #if there are too many vessels, all cells have 1 oxygen

            else:
                alpha_ = self.alpha_map.evaluate((pressure, n_vessels))
                beta_ = self.beta_map.evaluate((pressure, n_vessels))

                if alpha_ < 0 or beta_ < 0:
                    print('pressure', pressure, 'n_vessels', n_vessels)
                    print('alpha_', alpha_, 'beta_', beta_)

                o2_values = np.random.beta(alpha_, beta_, size=n_cells) #sample from beta distribution

        for i in range(n_cells):
            voxel.list_of_cells[i].oxygen = o2_values[i]

class UpdateVoxelMolecules(Process): #update the molecules in the voxel (VEGF), other not implemented yet
    def __init__(self, config, name, dt):
        super().__init__(config, 'UpdateMolecules', dt)

    def update_VEGF(self, voxel: Voxel): #update the VEGF concentration in the voxel
        VEGF = 0.0
        for cell in voxel.list_of_cells: #sum the VEGF secreted by each cell. Oxygen and and damage play a role in the secretion
            VEGF += cell.VEGF_secretion(self.config.metabolic_damage_threshold)
        VEGF = min(VEGF, 1.0)
        #print('New VEGF is',VEGF)
        voxel.molecular_factors['VEGF'] = VEGF
        return
    
    def update_fiber_density(self, voxel: Voxel): #update the fiber density in the voxel
        fiber_density = 0.0
        for cell in voxel.list_of_cells:
            fiber_density += cell.fiber_secretion
        fiber_density = min(fiber_density, 1.0)
        voxel.molecular_factors['fiber_density'] = fiber_density
        return

    @Process.timeit
    def __call__(self, voxel: Voxel):
        self.update_VEGF(voxel) #update the VEGF concentration
        # self.update_fiber_density(voxel)
        return
    
class UpdateVasculature(Process): #update the vasculature
    def __init__(self, config, name, dt, killing_radius_threshold, n_capillaries_per_VVD, capillary_length, splitting_rate, macro_steps, micro_steps, weight_direction, weight_vegf, weight_pressure):
        super().__init__(config, 'UpdateVasculature', dt)
        self.is_global = True
        self.killing_radius_threshold = killing_radius_threshold
        self.n_capillaries_per_VVD = n_capillaries_per_VVD
        self.capillary_length = capillary_length
        self.dt = dt
        self.splitting_rate = splitting_rate
        self.macro_steps = macro_steps
        self.micro_steps = micro_steps
        self.weight_direction = weight_direction
        self.weight_vegf = weight_vegf
        self.weight_pressure = weight_pressure

    @Process.timeit
    def __call__(self, world: World):
        #print in separate thread
        n_killed = world.vessels_killed(self.killing_radius_threshold) #kill vessels that have a radius smaller than the threshold

        print('Killed vessels: ', n_killed)
        print('Growing vessels')

        vessels = world.vasculature.list_of_vessels
        volume_world = 8*world.half_length**3
        n_new_vessels = int(self.config.new_vessels_per_hour * self.dt * volume_world)
        n_new_vessels = min(n_new_vessels, len(vessels)) #the number of new vessels cannot be larger than the number of existing vessels
        vegf = world.vegf_map(step_gradient= self.config.vegf_map_step_gradient) #compute the gradient of the VEGF map

        # figure = plt.figure()
        # ax = figure.add_subplot(111)
        # vegf.show_values(figure,ax, 'viridis', 0.0, 1.0)
        # plt.show()

        def vegf_gradient(point): return vegf.gradient(point) #define the gradient of the map

        for _ in range(n_new_vessels): #create a tEC on some vessels randomly
            random_vessel = random.choice(vessels)
            if len(random_vessel.path) > 2:
                point = random_vessel.choose_random_point(self.config.seed)
                if vegf.evaluate(point) > self.config.vegf_scalar_threshold: #if the VEGF concentration is too low, tEC does not start growing
                    if np.linalg.norm(vegf_gradient(point)) > self.config.vegf_gradient_threshold: #if the gradient of the VEGF map is large enough, tEC starts growing
                        world.vasculature.branching(random_vessel.id, point)

        #grow the vessels and update the volume occupied by the vessels
        #world.vasculature_growth(self.dt, self.splitting_rate, self.macro_steps, self.micro_steps, self.weight_direction, self.weight_vegf, self.weight_pressure)
        #world.update_volume_occupied_by_vessels()
        #update the capillary map
        #world.update_capillaries(n_capillaries_per_VVD= self.n_capillaries_per_VVD, capillary_length = self.capillary_length)

        world.vasculature_growth(self.dt, self.splitting_rate, self.macro_steps, self.micro_steps, self.weight_direction, self.weight_vegf, self.weight_pressure)

        world.update_volume_occupied_by_vessels()

        world.update_capillaries(n_capillaries_per_VVD= self.n_capillaries_per_VVD, capillary_length = self.capillary_length)


class Irradiation(Process): #irradiation
    def __init__(self, config, name, dt, topas_file, irradiation_intensity, world: World):
        super().__init__(config, 'Irradiation', dt)
        self.irradiation_intensity = irradiation_intensity

        #check if the file exists
        if not os.path.isfile(topas_file + '.csv'):
            #if it does not exist, run the simulation
            print('Running Topas simulation')
            print('Topas file: ', topas_file)
            file_with_geom = world.topas_param_file(topas_file)
            print('Topas param file: ', file_with_geom)
            term.RunTopasSimulation(file_with_geom, cluster=self.config.running_on_cluster)
            os.rename('MyScorer.csv', topas_file + '.csv')

        #read the dose from the file
        _, read_doses = rw.DoseOnWorld(topas_file + '.csv')

        #transform numpy aray into a list
        read_doses = read_doses.tolist()

        self.doses = np.zeros(len(read_doses)) #create an array of zeros

        #added to compensate for inhomogeneites in the Topas simulation, TODO: fix this in the Topas simulation and remove this
        for i in range(len(read_doses)):
            self.doses[i] = 0.5*(read_doses[i] + read_doses[len(read_doses) - i - 1])

        world.update_dose(self.doses) #update the dose on the world

        # plot the simulation
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        world.show_tumor_slice(ax[0], fig, 'dose', cmap='RdPu',refinement_level=1, slice='x', round_n=1)
        world.show_tumor_slice(ax[1], fig, 'dose', cmap='RdPu',refinement_level=1, slice='y', round_n=1)
        world.show_tumor_slice(ax[2], fig, 'dose', cmap='RdPu',refinement_level=1, slice='z', round_n=1)
        fig.suptitle('Dose (Gy)', fontsize=20)
        plt.tight_layout()
        plt.savefig('dose.png', dpi=300)
        if self.config.running_on_cluster:
            plt.close()
        else:
            plt.show()
    @Process.timeit
    def __call__(self, world: World):
        for voxel in world.voxel_list:
            scaled_dose = self.doses[voxel.voxel_number]*self.irradiation_intensity
            if len(voxel.list_of_cells) > 0:
                print('Scaled dose: ', scaled_dose)
                for cell in voxel.list_of_cells:
                    #assume all cells get damaged the same way
                    damage = scaled_dose * cell.radiosensitivity() #compute the damage
                    cell.damage += damage
                    cell.damage = min(cell.damage, 1.0)

        for vessel in world.vasculature.list_of_vessels:
            path = vessel.path
            total_dose = 0

            for point in path: #compute the mean dose on the vessel
                current_voxel = world.find_voxel_number(point)
                total_dose += world.voxel_list[current_voxel].dose

            if len(path) > 0:
                mean_dose = total_dose / len(path)
                vessel.must_be_updated = True
            else:
                mean_dose = 0

            damage_vessel = mean_dose * self.irradiation_intensity * vessel.radiosensitivity()
            vessel.maturity -= damage_vessel
            if vessel.maturity < 0:
                vessel.maturity = 0

        return
