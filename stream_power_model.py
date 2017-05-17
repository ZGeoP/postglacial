# This is a model that creates an idealized glaciated upland
# The upland has a moraine along one boundary, a flat upland
# and an incising channel along another boundary
# the landscape erodes by stream power with water routed over
# topography, and can routed over flats/pits.
# Modified by JL Sep. 2015

def run_model(input_file=None, savepath=None, initial_topo_file=None, initial_seed_file=None):
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.stream_power.fastscape_stream_power_JL import FastscapeEroder
    #from landlab.components.stream_power.stream_power import StreamPowerEroder
    from landlab.components.sink_fill.pit_fill_pf import PitFiller
    from landlab.components.diffusion.diffusion import LinearDiffuser
    from landlab import ModelParameterDictionary
    #from landlab.plot import channel_profile as prf
    from landlab.plot.imshow import imshow_node_grid
    from landlab.io.esri_ascii import write_esri_ascii
    from landlab.io.esri_ascii import read_esri_ascii
    from landlab import RasterModelGrid

    #from analysis_method import analyze_drainage_percentage
    #from analysis_method import analyze_drainage_percentage_each_grid
    #from analysis_method import analyze_mean_erosion
    #from analysis_method import elev_diff_btwn_moraine_upland
    #from analysis_method import label_catchment
    #from analysis_method import cross_section
    #from analysis_method import analyze_node_below_threshold
    #from analysis_method import identify_drained_area
    #from analysis_method import save_result
    from analysis_method import shiftColorMap

    import copy
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    import time
    import sys
    import shutil

    sys.setrecursionlimit(5000)

    #===============================================================================
    #get the needed properties to build the grid:
    if input_file is None:
        input_file = './coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    nrows = inputs.read_int('nrows')
    ncols = inputs.read_int('ncols')
    dx = inputs.read_float('dx')
    initial_slope = inputs.read_float('initial_slope')
    rightmost_elevation = initial_slope*ncols*dx
    #rightmost_elevation = inputs.read_float('rightmost_elevation')
    uplift_rate = inputs.read_float('uplift_rate')
    incision_rate = inputs.read_float('incision_rate')
    runtime = inputs.read_float('total_time')
    dt = inputs.read_float('dt')
    nt = int(runtime//dt)
    k_sp = inputs.read_float('K_sp')
    m_sp = inputs.read_float('m_sp')
    uplift_per_step = uplift_rate * dt
    incision_per_step = incision_rate * dt
    moraine_height = inputs.read_float('moraine_height')
    moraine_width = inputs.read_float('moraine_width')
    #valley_width = inputs.read_float('valley_width')
    valley_depth = inputs.read_float('valley_depth')
    num_outs = inputs.read_int('number_of_outputs')
    output_interval = int(nt//num_outs)
    diff = inputs.read_float('linear_diffusivity')
    #threshold_stream_power = inputs.read_float('threshold_stream_power')
    threshold_AS = inputs.read_float('threshold_AS')
    #threshold_erosion = dt*threshold_stream_power
    gw_coeff = inputs.read_float('gw_coeff')
    all_dry = inputs.read_bool('all_dry')
    fill_sink_with_water = inputs.read_bool('fill_sink_with_water')
    #===============================================================================

    #===============================================================================
    if initial_topo_file is None:
        #instantiate the grid object
        mg = RasterModelGrid(nrows, ncols, dx)

        ##create the elevation field in the grid:
        #create the field
        #specifically, this field has a triangular ramp
        #moraine at the north edge of the domain.
        mg.add_zeros('node', 'topographic__elevation', units='m')
        z = mg.at_node['topographic__elevation']
        moraine_start_y = np.max(mg.node_y)-moraine_width
        moraine_ys = np.where(mg.node_y>moraine_start_y)
        z[moraine_ys]+=(mg.node_y[moraine_ys]-np.min(mg.node_y[moraine_ys]))*(moraine_height/moraine_width)

        #set valley
        #valley_start_x = np.min(mg.node_x)+valley_width
        #valley_ys = np.where((mg.node_x<valley_start_x)&(mg.node_y<moraine_start_y-valley_width))
        #z[valley_ys] -= (np.max(mg.node_x[valley_ys])-mg.node_x[valley_ys])*(valley_depth/valley_width)

        #set ramp (towards valley)
        upland = np.where(mg.node_y<moraine_start_y)
        z[upland] -= (np.max(mg.node_x[upland])-mg.node_x[upland])*(rightmost_elevation/(ncols*dx))
        z += rightmost_elevation

        #set ramp (away from moraine)
        #upland = np.where(mg.node_y<moraine_start_y)
        #z[upland] -= (moraine_start_y-mg.node_y[upland])*initial_slope

        #put these values plus roughness into that field
        if initial_seed_file is None:
            z += np.random.rand(len(z))/1
        else:
            (seedgrid, seed) = read_esri_ascii(initial_seed_file, name = 'topographic__elevation_seed')
            z += seed
        mg.at_node['topographic__elevation'] = z

        #set river valley
        river_valley, = np.where(np.logical_and(mg.node_x==0, np.logical_or(mg.status_at_node==1, mg.status_at_node==2)))
        mg.at_node['topographic__elevation'][river_valley] = -valley_depth
    else:
        (mg, z) = read_esri_ascii(initial_topo_file, name = 'topographic__elevation')

    #set river valley
    river_valley, = np.where(np.logical_and(mg.node_x==0, np.logical_or(mg.status_at_node==1, mg.status_at_node==2)))
    mg.at_node['topographic__elevation'][river_valley] = -valley_depth

    #set up grid's boundary conditions (right, top, left, bottom) is inactive
    mg.set_closed_boundaries_at_grid_edges(True, True, False, True)

    #set up boundary along moraine
    #moraine_start_y = np.max(mg.node_y)-moraine_width
    #bdy_moraine_ids = np.where((mg.node_y > moraine_start_y) & (mg.node_x == 0))
    #mg.status_at_node[bdy_moraine_ids]=4
    #mg._update_links_nodes_cells_to_new_BCs()
    #===============================================================================

    #===============================================================================
    #instantiate the components:
    fr = FlowRouter(mg)
    pf = PitFiller(mg)
    sp = FastscapeEroder(mg, input_file, threshold_sp = threshold_AS*k_sp)
    #sp = StreamPowerEroder(mg, input_file, threshold_sp=threshold_erosion, use_Q=True)
    #diffuse = PerronNLDiffuse(mg, input_file)
    #lin_diffuse = LinearDiffuser(mg, input_file, method='on_diagonals')
    lin_diffuse = LinearDiffuser(mg, input_file)
    #===============================================================================

    #===============================================================================
    #instantiate plot setting
    plt.close('all')
    output_time = output_interval
    plot_num = 0
    mycmap = shiftColorMap(matplotlib.cm.gist_earth, 'mycmap')

    #folder name
    if savepath is None:
        if all_dry:
            name_tag = 'All_dry'
        elif fill_sink_with_water:
            name_tag = 'Not_all_dry_no_sink'
        else:
            name_tag = 'Not_all_dry_sink'
        savepath = 'results/sensitivity_test_threshold_same_seed/'+name_tag+'_dt=' + str(dt) + '_total_time=' + str(runtime) + '_k_sp=' + str(k_sp) + \
                '_uplift_rate=' + str(uplift_rate) + '_incision_rate=' + str(incision_rate) + '_initial_slope=' + str(initial_slope) + \
                '_threshold=' + str(threshold_stream_power)
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    #copy params_file
    if not os.path.isfile(savepath+'/coupled_params_sp.txt'):
        shutil.copy(input_file, savepath+'/coupled_params_sp.txt')

    # Display a message
    print 'Running ...'
    print savepath

    #save initial topography
    write_esri_ascii(savepath + '/Topography_t=0.0.txt', mg, 'topographic__elevation')

    #time
    start_time = time.time()
    #===============================================================================

    #===============================================================================
    #perform the loops:
    for i in xrange(nt):
        #note the input arguments here are not totally standardized between modules

        ''' 
        # simulate changing climate
        if (((i+1)*dt // 5000.) % 2) == 0.:
            all_dry = False
        else:
            all_dry = True
        '''

        #sp = FastscapeEroder(mg, input_file, threshold_sp = threshold_stream_power)

        #update elevation
        mg.at_node['topographic__elevation'][mg.core_nodes] += uplift_per_step
        mg.at_node['topographic__elevation'][river_valley] -= incision_per_step

        #mg = lin_diffuse.diffuse(dt)

        #route and erode
        original_topo = mg.at_node['topographic__elevation'].copy()
        slope = mg.at_node['topographic__steepest_slope'].copy()
        filled_nodes = None
        if all_dry or fill_sink_with_water:
            mg = pf.pit_fill()
            filled_nodes, = np.where((mg.at_node['topographic__elevation']-original_topo)>0)
        mg = fr.route_flow(routing_flat=all_dry)
        old_topo = mg.at_node['topographic__elevation'].copy()
        #mg, temp_z, temp_sp = sp.erode(mg, dt, Q_if_used='water__volume_flux')
        mg = sp.erode(mg, dt, flooded_nodes=filled_nodes)

        new_topo = mg.at_node['topographic__elevation'].copy()
        mg.at_node['topographic__elevation'] = original_topo + new_topo - old_topo

        #diffuse
        #for j in range(10):
        #    mg = lin_diffuse.diffuse(dt/10)
        mg = lin_diffuse.diffuse(dt)

        if i+1 == output_time:
            print 'Saving data...'

            plot_num += 1
            plt.figure(plot_num)
            im = imshow_node_grid(mg, 'topographic__elevation', plot_name='t = '+str(int((i+1)*dt))+' years', \
                grid_units = ['m','m'], cmap=mycmap, allow_colorbar=True, \
                vmin=0-valley_depth-incision_rate*runtime, vmax=5.+moraine_height+uplift_rate*runtime)
            plt.savefig(savepath + '/Topography_t=' + str((i+1)*dt) + '.jpg', dpi = 300)

            write_esri_ascii(savepath + '/Topography_t=' + str((i+1)*dt) + '.txt', mg, 'topographic__elevation')

            output_time += output_interval

        plt.close('all')

        print ("--- %.2f minutes ---" % ((time.time() - start_time)/60)), 'Completed loop', i+1

    plt.close('all')

    print 'Finished simulating.'
    #===============================================================================

    #===============================================================================
    #analyze_drainage_percentage(savepath, True)
    #analyze_mean_erosion(savepath, True)
    #elev_diff_btwn_moraine_upland(savepath, True)
    #label_catchment(savepath)
    #cross_section(savepath)
    #===============================================================================

    print 'Done!'
    print '\n'

if __name__ == '__main__':
    run_model()
