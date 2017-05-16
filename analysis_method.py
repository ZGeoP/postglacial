from landlab.io.esri_ascii import read_esri_ascii
from landlab.io.esri_ascii import write_esri_ascii
from landlab import RasterModelGrid
from landlab import ModelParameterDictionary

import numpy as np
import matplotlib.pyplot as plt


def shiftColorMap(cmap, name, midpoint=0.9):
    import matplotlib
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    # regular index to compute the colors
    reg_index = np.linspace(0, 1, 257)

    #shift_index = np.power(reg_index, 1./3.)

    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])


    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap


def channel_profile():

    from landlab.plot import channel_profile as prf
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter

    plt.figure(1)
    #pdb.set_trace()
    (mg, z) = read_esri_ascii('profile1.txt', name = 'topographic__elevation')
    mg.set_closed_boundaries_at_grid_edges(True, False, True, True)
    fr = FlowRouter(mg)
    mg = fr.route_flow(routing_flat=True)
    temp = mg.diagonal_links_at_node()

    profile_IDs = prf.channel_nodes(mg, mg.at_node['topographic__steepest_slope'],
            mg.at_node['drainage_area'], mg.at_node['flow_receiver'])
    dists_upstr = prf.get_distances_upstream(mg, len(mg.at_node['topographic__steepest_slope']),
            profile_IDs, mg.at_node['links_to_flow_receiver'])
    prf.plot_profiles(dists_upstr, profile_IDs, mg.at_node['topographic__elevation'])


    (mg, z) = read_esri_ascii('profile2.txt', name = 'topographic__elevation')
    mg.set_closed_boundaries_at_grid_edges(True, False, True, True)
    fr = FlowRouter(mg)
    mg = fr.route_flow(routing_flat=True)
    temp = mg.diagonal_links_at_node()

    profile_IDs = prf.channel_nodes(mg, mg.at_node['topographic__steepest_slope'],
            mg.at_node['drainage_area'], mg.at_node['flow_receiver'])
    dists_upstr = prf.get_distances_upstream(mg, len(mg.at_node['topographic__steepest_slope']),
            profile_IDs, mg.at_node['links_to_flow_receiver'])
    prf.plot_profiles(dists_upstr, profile_IDs, mg.at_node['topographic__elevation'])

    plt.savefig('channel_profile.jpg')

    plt.close('all')


def plot_grid_3d(file_name, zticks=None):

    from mpl_toolkits.mplot3d import Axes3D

    (mg, z) = read_esri_ascii(file_name+'.txt', name = 'topographic__elevation')

    z = mg.at_node['topographic__elevation'][mg.core_nodes]
    z = z.reshape(mg.cell_grid_shape)
    y = np.arange(mg.cell_grid_shape[0])*mg.dx
    x = np.arange(mg.cell_grid_shape[1])*mg.dy
    x, y = np.meshgrid(x, y)

    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.view_init(azim=-120)
    #ax.invert_xaxis()

    '''
    x_scale=1
    y_scale=1
    z_scale=0.2

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj=short_proj
    '''
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='jet', linewidth=0)

    #ax.set_zticks([-50, 20])
    if not(zticks is None):
        ax.set_zticks(zticks)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig(file_name+'_3d.jpg', dpi=300)
    plt.close('all')


def rewrite_params_file(file_name, kwd, value):

    param_file = file(file_name, 'r')
    line_list = []
    for line in param_file:
        line = line.strip()
        if len(line) > 0:
            line_list.append(line)
    param_file.close()
    for i in range(len(line_list)):
        line = line_list[i]
        first_colon = line.find(':')
        if first_colon == -1:
            first_colon = len(line)
        first_space = line.find(' ')
        if first_space == -1:
            first_space = len(line)
        key_char = min(first_colon, first_space)
        key = line[0:key_char]
        if key == kwd:
            line_list[i+1] = value
            break

    param_file = file(file_name, 'w')
    for line in line_list:
        param_file.write(line+'\n')
    param_file.close()


def iteratively_run(kwd, value_list, input_file=None):

    from stream_power_model import run_model
    if input_file is None:
        input_file = './coupled_params_sp.txt'
    for value in value_list:
        rewrite_params_file(input_file, kwd, value)
        run_model(input_file=input_file)


def save_result(t, result, filepath):

    output = file(filepath, 'w')
    output.write("%s\n" % len(t))
    for i in t:
        output.write("%s\n" % i)
    for i in result:
        output.write("%s\n" % i)
    output.close()


def build_model_grid(path, dem_file_name):

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    moraine_width = inputs.read_float('moraine_width')

    (mg, z) = read_esri_ascii(path+dem_file_name, name = 'topographic__elevation')
    mg.set_closed_boundaries_at_grid_edges(True, True, False, True)
    '''
    moraine_start_y = np.max(mg.node_y)-moraine_width
    bdy_moraine_ids = np.where((mg.node_y > moraine_start_y) & (mg.node_x == 0))
    mg.status_at_node[bdy_moraine_ids]=4
    mg._update_links_nodes_cells_to_new_BCs()
    '''

    return mg


def identify_drained_area(grid):

    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter

    fr = FlowRouter(grid)
    grid = fr.route_flow(routing_flat=False)
    node_stack = grid.at_node['flow__upstream_node_order']
    receiver = grid.at_node['flow__receiver_node']
    open_boundary, = np.where(np.logical_or(grid.status_at_node==1, grid.status_at_node==2))

    n = grid.number_of_nodes
    flow_to_boundary = np.zeros(n, dtype=bool)
    for i in open_boundary:
        flow_to_boundary[i] = True

    for i in range(n):
        flow_to_boundary[node_stack[i]] = flow_to_boundary[receiver[node_stack[i]]]

    return flow_to_boundary


def analyze_drainage_percentage_each_grid(grid):
    flow_to_boundary = identify_drained_area(grid)
    count, = np.where(flow_to_boundary)
    count = len(count)
    count = float(count)
    percentage = count/(grid.number_of_nodes)

    return percentage


def analyze_drainage_percentage(path, plotting=True):

    print 'Analysing drainage percentage...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    percentage = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        mg.at_node['flow_to_boundary'] = identify_drained_area(mg)
        flow_to_boundary = mg.at_node['flow_to_boundary']
        count, = np.where(flow_to_boundary)
        count = len(count)
        count = float(count)
        nrows, ncols = mg.shape
        percentage[i] = count/(nrows*ncols)
        print 'Finished loop', i+1

    save_result(t, percentage, path+'/drainage_percentage.txt')

    if plotting:
        plt.figure(1)
        plt.plot(t, percentage)
        plt.ylim([0,1])
        plt.xlabel('Time (years)')
        plt.ylabel('Drainage Percentage')
        plt.savefig(path+'/drainage_percentage.jpg', dpi=300)
        plt.close('all')

    #print '\n'
    #return t, percentage

'''
def analyze_drainage_percentage_temp(path, plotting=True):
    from landlab.components.stream_power.fastscape_stream_power_JL import FastscapeEroder
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller

    print 'Analysing drainage percentage...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    percentage = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        sp = FastscapeEroder(mg, K_sp = 1e-04)
        fr = FlowRouter(mg)
        pf = PitFiller(mg)
        mg = pf.pit_fill()
        mg = fr.route_flow(routing_flat=True)
        mg = sp.erode(mg, 100.)
        mg.at_node['flow_to_boundary'] = identify_drained_area(mg)
        flow_to_boundary = mg.at_node['flow_to_boundary']
        count, = np.where(flow_to_boundary)
        count = len(count)
        count = float(count)
        nrows, ncols = mg.shape
        percentage[i] = count/(nrows*ncols)
        print 'Finished loop', i+1

    save_result(t, percentage, path+'/drainage_percentage.txt')

    if plotting:
        plt.figure(1)
        plt.plot(t, percentage)
        plt.ylim([0,1])
        plt.xlabel('Time (years)')
        plt.ylabel('Drainage Percentage')
        plt.savefig(path+'/drainage_percentage.jpg', dpi=300)
        plt.close('all')
'''

def identify_drained_area_elev_thr(grid, threshold_elevation):

    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter

    fr = FlowRouter(grid)
    grid = fr.route_flow(routing_flat=False)
    node_stack = grid.at_node['flow__upstream_node_order']
    receiver = grid.at_node['flow__receiver_node']
    open_boundary, = np.where(np.logical_or(np.logical_or(grid.status_at_node==1, grid.status_at_node==2), \
                              grid.at_node['topographic__elevation']<threshold_elevation))

    n = grid.number_of_nodes
    flow_to_boundary = np.zeros(n, dtype=bool)
    for i in open_boundary:
        flow_to_boundary[i] = True

    for i in range(n):
        flow_to_boundary[node_stack[i]] = flow_to_boundary[receiver[node_stack[i]]]

    fr = 1.

    return flow_to_boundary


def analyze_drainage_percentage_elev_thr(path, plotting=True):
    from landlab.plot.imshow import imshow_node_grid

    print 'Analysing drainage percentage...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    initial_mg = build_model_grid(path, '/Topography_t=0.0.txt')
    min_elev = np.nanmin(initial_mg.at_node['topographic__elevation'][initial_mg.core_nodes])

    t = np.zeros(num_outs)
    percentage = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        mg.at_node['flow_to_boundary'] = identify_drained_area_elev_thr(mg, threshold_elevation=min_elev-0.01)
        flow_to_boundary = mg.at_node['flow_to_boundary']
        count, = np.where(flow_to_boundary)
        count = len(count)
        count = float(count)
        nrows, ncols = mg.shape
        percentage[i] = count/(nrows*ncols)

        plt.close('all')
        plt.figure(1)
        im = imshow_node_grid(mg, 'flow_to_boundary', plot_name='t = '+str(int((i+1)*dt))+' years', \
                              grid_units = ['m','m'], cmap='Greys', allow_colorbar=False)
        plt.savefig(path + '/Drainage_of_valley_t=' + str((i+1)*dt) + '.jpg', dpi = 300)
        plt.close('all')

        mg = 1.

        print 'Finished loop', i+1

    save_result(t, percentage, path+'/drainage_percentage.txt')

    if plotting:
        plt.figure(1)
        plt.plot(t, percentage)
        plt.ylim([0,1])
        plt.xlabel('Time (years)')
        plt.ylabel('Drainage Percentage')
        plt.savefig(path+'/drainage_percentage.jpg', dpi=300)
        plt.close('all')


def plot_result(file_name, mkr='v'):

    inputfile = file(file_name, 'r')
    n = int(inputfile.readline())
    t = np.zeros(n, dtype=float)
    result = np.zeros(n, dtype=float)
    for i in range(n):
        t[i] = float(inputfile.readline())
    for i in range(n):
        result[i] = float(inputfile.readline())
    inputfile.close()
    plt.plot(t, result, marker=mkr)


def plot_result_legend(file_name, kwd, value, mkr='v'):

    inputfile = file(file_name, 'r')
    n = int(inputfile.readline())
    t = np.zeros(n, dtype=float)
    result = np.zeros(n, dtype=float)
    for i in range(n):
        t[i] = float(inputfile.readline())
    for i in range(n):
        result[i] = float(inputfile.readline())
    inputfile.close()
    if kwd=='initial_slope':
        kwd = 'Initial slope'
    if kwd=='k_sp':
        kwd = 'K'
    if kwd=='incision_rate':
        kwd = 'Incision rate'
    if kwd=='threshold_stream_power':
        kwd = 'Threshold value'
    plt.plot(t, result, marker=mkr, label=kwd+'='+str(value))


def batch_analysis(kwd, value_list, input_file=None):

    if input_file is None:
        input_file = './coupled_params_sp.txt'

    for value in value_list:
        rewrite_params_file(input_file, kwd, value)
        savepath = generate_savepath(input_file=input_file)
        analyze_drainage_percentage(path=savepath)


def analyze_mean_erosion(path, plotting=True):

    print 'Analysing mean erosion...'
    print path

    params_file = ModelParameterDictionary(path+'/coupled_params_sp.txt')
    uplift_rate = params_file.read_float('uplift_rate')
    initial_mg = build_model_grid(path, '/Topography_t=0.0.txt')
    num_outs = params_file.read_int('number_of_outputs')
    runtime = params_file.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    mean_erosion = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        erosion = initial_mg.at_node['topographic__elevation'][mg.core_nodes]+uplift_rate*t[i]-mg.at_node['topographic__elevation'][mg.core_nodes]
        mean_erosion[i] = np.mean(erosion)
        print 'Finished loop', i+1

    save_result(t, mean_erosion, path+'/mean_erosion.txt')

    if plotting:
        plt.figure(1)
        plt.plot(t, mean_erosion)
        plt.xlabel('Time (years)')
        plt.ylabel('Mean Erosion (m)')
        plt.savefig(path+'/mean_erosion.jpg')
        plt.close('all')


def elev_diff_btwn_moraine_upland(path, plotting=True):

    print 'Analysing elevation difference...'
    print path

    params_file = ModelParameterDictionary(path+'/coupled_params.txt')
    moraine_width = params_file.read_float('moraine_width')
    mg = build_model_grid(path, '/Topography_t=0.0.txt')
    moraine_start_y = np.max(mg.node_y)-moraine_width
    moraine, = np.where(mg.node_y>moraine_start_y)
    upland, = np.where(mg.node_y<=moraine_start_y)
    moraine = np.intersect1d(moraine, mg.core_nodes)
    upland = np.intersect1d(upland, mg.core_nodes)

    num_outs = params_file.read_int('number_of_outputs')
    runtime = params_file.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    diff = np.zeros(num_outs)

    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        diff[i] = np.mean(mg.at_node['topographic__elevation'][moraine])-np.mean(mg.at_node['topographic__elevation'][upland])
        print 'Finished loop', i+1

    save_result(t, diff, path+'/elev_diff_btwn_moraine_upland.txt')

    if plotting:
        plt.figure(1)
        plt.plot(t, diff)
        #plt.ylim([-5, 50])
        plt.xlabel('Time (years)')
        plt.ylabel('Elevation Difference between Moraine and Upland (m)')
        plt.savefig(path+'/elev_diff_btwn_moraine_upland.jpg')
        plt.close('all')


def analyze_percentage_of_sink(path, plotting=True):

    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    percentage = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        fr = FlowRouter(mg)
        mg = fr.route_flow(routing_flat=False)
        sink, = np.where(mg.at_node['flow_sinks'])
        num_sink = len(sink)
        num_sink = float(num_sink)
        nrows, ncols = mg.shape
        percentage[i] = num_sink/(nrows*ncols)
        print 'Finished loop', i+1

    save_result(t, percentage, path+'/percentage_of_sinks.txt')

    if plotting:
        plt.figure(1)
        plt.plot(t, percentage)
        plt.ylim([0,1])
        plt.xlabel('Time (years)')
        plt.ylabel('Percentage of Sinks')
        plt.savefig(path+'/percentage_of_sinks.jpg')
        plt.close('all')


def analyze_percentage_of_sink_in_upland(path, plotting=True):

    print 'Analysing percentage of sink in upland...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    percentage = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        mg.at_node['flow_to_boundary'] = identify_drained_area(mg)
        flow_to_boundary = mg.at_node['flow_to_boundary']
        count, = np.where(flow_to_boundary)
        count = len(count)
        count = float(count)
        sink, = np.where(np.logical_and(mg.at_node['flow_sinks'], mg.status_at_node==0))
        num_sink = len(sink)
        num_sink = float(num_sink)
        nrows, ncols = mg.shape
        percentage[i] = num_sink/(nrows*ncols-count)
        print 'Finished loop', i+1

    save_result(t, percentage, path+'/percentage_of_sinks_in_upland.txt')

    if plotting:
        plt.figure(1)
        plt.plot(t, percentage)
        plt.ylim([0,1])
        plt.xlabel('Time (years)')
        plt.ylabel('Percentage of Sinks')
        plt.savefig(path+'/percentage_of_sinks_in_upland.jpg')
        plt.close('all')


def analyze_drainage_of_sink(path):

    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        fr = FlowRouter(mg)
        mg = fr.route_flow(routing_flat=False)
        sink, = np.where(np.logical_and(mg.at_node['flow__sink_flag'], mg.status_at_node==0))
        area = mg.at_node['drainage_area'][sink]
        plt.figure(1)
        plt.hist(area, 20)
        plt.xlabel('Drainage Area')
        plt.savefig(path+'/hist_drainage_of_sink_t='+str(t[i])+'.jpg')
        plt.close('all')
        print 'Finished loop', i+1


def analyze_mean_drainage_of_sink(path, plotting=True):

    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter

    print 'Analysing mean drainage of sink...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs+1)
    mean_drainage = np.zeros(num_outs+1)
    for i in range(num_outs+1):
        t[i] = i*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        fr = FlowRouter(mg)
        mg = fr.route_flow(routing_flat=False)
        sink, = np.where(np.logical_and(mg.at_node['flow__sink_flag'], mg.status_at_node==0))
        area = mg.at_node['drainage_area'][sink]
        if len(area)!=0:
            mean_drainage[i] = np.mean(area)
        else:
            mean_drainage[i] = 0.
        print 'Finished loop', i+1

    save_result(t, mean_drainage, path+'/mean_drainage_of_sink.txt')

    if plotting:
        plt.close('all')
        plt.figure(1)
        plt.plot(t, mean_drainage)
        #plt.ylim([0,1])
        plt.xlim([0, runtime])
        plt.xlabel('Time (years)')
        plt.ylabel('Mean Drainage of Sinks (m*m)')
        plt.savefig(path+'/mean_drainage_of_sink.jpg', dpi=300)
        plt.close('all')


def label_catchment_each_grid(grid):

    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter

    fr = FlowRouter(grid)
    grid = fr.route_flow(routing_flat=False)
    node_stack = grid.at_node['flow__upstream_node_order']
    receiver = grid.at_node['flow__receiver_node']
    open_boundary, = np.where(np.logical_or(grid.status_at_node==1, grid.status_at_node==2))
    sink, = np.where(np.logical_and(grid.at_node['flow__sink_flag'], grid.status_at_node==0))

    n = grid.number_of_nodes
    catchment_ID = np.zeros(n, dtype=int)
    for i in range(n):
        catchment_ID[i] = i
    for i in open_boundary:
        catchment_ID[i] = -1

    ID_list = np.zeros(len(sink), dtype=int)
    for i in range(len(ID_list)):
        ID_list[i] = i

    n_id = len(ID_list)-1
    for node in sink:
        k = np.random.random_integers(0, n_id)
        catchment_ID[node] = ID_list[k]
        temp = ID_list[k]
        ID_list[k] = ID_list[n_id]
        ID_list[n_id] = temp
        n_id -= 1

    for i in range(n):
        catchment_ID[node_stack[i]] = catchment_ID[receiver[node_stack[i]]]

    grid.at_node['catchment_ID'] = catchment_ID

    return grid


def label_catchment(path):

    from landlab.plot.imshow import imshow_node_grid

    print 'Labeling catchment...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        mg = label_catchment_each_grid(mg)
        plt.figure(1)
        im = imshow_node_grid(mg, 'catchment_ID', plot_name='t = '+str(int((i+1)*dt))+' years', \
                              grid_units = ['m','m'], cmap='gist_earth', allow_colorbar=False)
        plt.savefig(path + '/catchment_ID_t=' + str(t[i]) + '.jpg', dpi=300)
        plt.close('all')
        print 'Finished loop', i+1


def label_catchment_each_grid_route(grid):

    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller

    fr = FlowRouter(grid)
    pf = PitFiller(grid)
    grid = pf.pit_fill()
    grid = fr.route_flow(routing_flat=True)
    node_stack = grid.at_node['flow__upstream_node_order']
    receiver = grid.at_node['flow__receiver_node']
    open_boundary, = np.where(np.logical_or(grid.status_at_node==1, grid.status_at_node==2))

    n = grid.number_of_nodes
    catchment_ID = np.zeros(n, dtype=int)
    for i in range(n):
        catchment_ID[i] = i

    ID_list = np.zeros(len(open_boundary), dtype=int)
    for i in range(len(ID_list)):
        ID_list[i] = i

    n_id = len(ID_list)-1
    for node in open_boundary:
        k = np.random.random_integers(0, n_id)
        catchment_ID[node] = ID_list[k]
        temp = ID_list[k]
        ID_list[k] = ID_list[n_id]
        ID_list[n_id] = temp
        n_id -= 1

    for i in range(n):
        catchment_ID[node_stack[i]] = catchment_ID[receiver[node_stack[i]]]

    grid.at_node['catchment_ID_route'] = catchment_ID

    return grid


def label_catchment_route(path):

    from landlab.plot.imshow import imshow_node_grid

    print 'Labeling catchment...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        mg = label_catchment_each_grid_route(mg)
        plt.figure(1)
        im = imshow_node_grid(mg, 'catchment_ID_route', plot_name='t = '+str(int((i+1)*dt))+' years', \
                              grid_units = ['m','m'], cmap='gist_earth', allow_colorbar=False)
        plt.savefig(path + '/catchment_ID_route_t=' + str(t[i]) + '.jpg', dpi=300)
        plt.close('all')
        print 'Finished loop', i+1


def analyze_depth_of_sink_each_grid(grid):

    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller

    fr = FlowRouter(grid)
    pf = PitFiller(grid)
    depth = np.zeros(grid.number_of_nodes)

    grid = fr.route_flow(routing_flat=False)
    sink, = np.where(np.logical_and(grid.at_node['flow__sink_flag'], grid.status_at_node==0))
    node_stack = grid.at_node['flow__upstream_node_order']
    receiver = grid.at_node['flow__receiver_node']

    original_topo = grid.at_node['topographic__elevation'].copy()
    grid = pf.pit_fill()
    depth[sink] = grid.at_node['topographic__elevation'][sink]-original_topo[sink]
    mean_depth = np.mean(depth[sink])

    for i in range(len(node_stack)):
        depth[node_stack[i]] = depth[receiver[node_stack[i]]]

    grid.at_node['sink_depth'] = depth

    return grid, mean_depth


def analyze_depth_of_sink(path, plotting=True):

    from landlab.plot.imshow import imshow_node_grid

    print 'Analyzing depth of each sinks...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs+1)
    mean_sink_depth = np.zeros(num_outs+1)
    for i in range(num_outs+1):
        t[i] = i*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        mg, mean_sink_depth[i] = analyze_depth_of_sink_each_grid(mg)
        plt.figure(1)
        im = imshow_node_grid(mg, 'sink_depth', plot_name='t = '+str(int((i+1)*dt))+' years', \
                              grid_units = ['m','m'], cmap='jet', allow_colorbar=True)
        plt.savefig(path + '/sink_depth_t=' + str(t[i]) + '.jpg', dpi=300)
        plt.close('all')
        print 'Finished loop', i+1

    save_result(t, mean_sink_depth, path+'/mean_depth_of_sink.txt')

    if plotting:
        plt.close('all')
        plt.figure(1)
        plt.plot(t, mean_sink_depth)
        #plt.ylim([0,1])
        plt.xlim([0, runtime])
        plt.xlabel('Time (years)')
        plt.ylabel('Mean Depth of Sinks (m)')
        plt.savefig(path+'/mean_depth_of_sink.jpg', dpi=300)
        plt.close('all')


def analyze_volume_over_area_of_sink_each_grid(grid):
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller

    fr = FlowRouter(grid)
    pf = PitFiller(grid)

    grid = fr.route_flow(routing_flat=False)
    sink, = np.where(np.logical_and(grid.at_node['flow__sink_flag'], grid.status_at_node==0))
    node_stack = grid.at_node['flow__upstream_node_order']
    receiver = grid.at_node['flow__receiver_node']

    original_topo = grid.at_node['topographic__elevation'].copy()
    grid = pf.pit_fill()
    filled_topo = grid.at_node['topographic__elevation'].copy()

    catchment_ID = np.zeros(grid.number_of_nodes)
    volume = np.zeros(len(sink))
    area = np.zeros(len(sink))

    for i in range(len(sink)):
        catchment_ID[sink[i]] = i
        area[i] = grid.at_node['drainage_area'][sink[i]]

    for i in range(len(node_stack)):
        catchment_ID[node_stack[i]] = catchment_ID[receiver[node_stack[i]]]

    for node in grid.core_nodes:
        volume[catchment_ID[node]] += (filled_topo[node]-original_topo[node])*grid.dx*grid.dx

    volume_over_area = volume / area
    mean_volume_over_area = np.mean(volume_over_area)

    return grid, mean_volume_over_area


def analyze_volume_over_area_of_sink(path, plotting=True):
    from landlab.plot.imshow import imshow_node_grid

    print 'Analyzing volume over area of each sinks...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs+1)
    mean_volume_over_area = np.zeros(num_outs+1)
    for i in range(num_outs+1):
        t[i] = i*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        mg, mean_volume_over_area[i] = analyze_volume_over_area_of_sink_each_grid(mg)
        print 'Finished loop', i+1

    save_result(t, mean_volume_over_area, path+'/mean_volume_over_area_of_sink.txt')

    if plotting:
        plt.close('all')
        plt.figure(1)
        plt.plot(t, mean_volume_over_area)
        #plt.ylim([0,1])
        plt.xlim([0, runtime])
        plt.xlabel('Time (years)')
        plt.ylabel('Mean Volume over Area of Sinks (m)')
        plt.savefig(path+'/mean_volume_over_area_of_sink.jpg', dpi=300)
        plt.close('all')



def cross_section(path):

    from landlab.plot.imshow import imshow_node_grid


    print 'Drawing cross section...'
    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    moraine_width = inputs.read_float('moraine_width')
    valley_depth = inputs.read_float('valley_depth')
    ncols = inputs.read_int('ncols')
    dx = inputs.read_float('dx')

    t = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        gw = np.array(range(ncols))
        gw = gw/float(ncols-1)*25.
        gw -= valley_depth
        z = mg.at_node['topographic__elevation']
        moraine_start_y = np.max(mg.node_y)-moraine_width
        cs = np.zeros(ncols)
        for j in range(ncols):
            cs[j] = np.mean(z[np.where((mg.node_y<moraine_start_y)&(mg.node_x==j*dx))])
        plt.figure(1)
        plt.plot(range(ncols), cs)
        plt.plot(range(ncols), gw)
        plt.savefig(path + '/cross_section_t=' + str(t[i]) + '.jpg')
        plt.close('all')
        mg.add_zeros('node', 'groundwater_level', units='m')
        mg.at_node['groundwater_level'] += mg.node_x/np.max(mg.node_x)*25
        mg.at_node['groundwater_level'] -= valley_depth
        lower_than_gw = np.zeros(len(z), dtype=bool)
        lower_than_gw[np.where(z<=mg.at_node['groundwater_level'])] = True
        mg.at_node['lower_than_gw'] = lower_than_gw
        im = imshow_node_grid(mg, 'lower_than_gw', var_name='t='+str((i+1)*dt), var_units='years', grid_units = ['m','m'], \
                              cmap='Greys', allow_colorbar=False)
        plt.savefig(path + '/lower_than_gw_t=' + str(t[i]) + '.jpg')
        plt.close('all')

        print 'Finished loop', i+1


def generate_savepath(folder_name, input_file=None):

    from landlab import ModelParameterDictionary

    if input_file is None:
        input_file = './coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    k_sp = inputs.read_float('K_sp')
    runtime = inputs.read_float('total_time')
    dt = inputs.read_float('dt')
    initial_slope = inputs.read_float('initial_slope')
    incision_rate = inputs.read_float('incision_rate')
    uplift_rate = inputs.read_float('uplift_rate')
    threshold_stream_power = inputs.read_float('threshold_stream_power')
    all_dry = inputs.read_bool('all_dry')
    fill_sink_with_water = inputs.read_bool('fill_sink_with_water')

    if all_dry:
        name_tag = 'All_dry'
    elif fill_sink_with_water:
        name_tag = 'Not_all_dry_no_sink'
    else:
        name_tag = 'Not_all_dry_sink'
    savepath = folder_name+name_tag+'_dt=' + str(dt) + '_total_time=' + str(runtime) + '_k_sp=' + str(k_sp) + \
            '_uplift_rate=' + str(uplift_rate) + '_incision_rate=' + str(incision_rate) + '_initial_slope=' + str(initial_slope) + \
            '_threshold=' + str(threshold_stream_power)

    return savepath


def find_node_below_threshold(path, grid):

    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    all_dry = inputs.read_bool('all_dry')
    k_sp = inputs.read_float('K_sp')
    m_sp = inputs.read_float('m_sp')
    threshold = inputs.read_float('threshold_stream_power')

    fr = FlowRouter(grid)
    pf = PitFiller(grid)
    if all_dry:
        grid = pf.pit_fill()
    grid = fr.route_flow(routing_flat=all_dry)
    a = grid.at_node['drainage_area']
    s = grid.at_node['topographic__steepest_slope']
    sp = k_sp*(a**m_sp)*s
    grid.at_node['lower_than_threshold'] = np.zeros(grid.number_of_nodes)
    grid.at_node['lower_than_threshold'][np.where(sp<=threshold)] = 1

    fr = 1.
    pf = 1.

    return grid


def analyze_node_below_threshold(path):

    from landlab.plot.imshow import imshow_node_grid

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    percentage = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        mg = find_node_below_threshold(path, mg)
        plt.close('all')
        plt.figure(1)
        im = imshow_node_grid(mg, 'lower_than_threshold', plot_name='t = '+str(int((i+1)*dt))+' years', \
                              grid_units = ['m','m'], cmap='Greys', allow_colorbar=False)
        plt.savefig(path + '/lower_than_threshold_t=' + str(t[i]) + '.jpg', dpi=300)
        plt.close('all')
        below_threshold = mg.at_node['lower_than_threshold'][mg.core_nodes]
        count = below_threshold.sum()
        percentage[i] = 1.0-float(count)/len(mg.core_nodes)

        mg = 1.
        print 'Finished loop', i+1

    save_result(t, percentage, path+'/above_threshold_percentage.txt')
    plt.close('all')
    plt.figure(1)
    plt.plot(t, percentage)
    plt.ylim([0,1])
    plt.xlabel('Time (years)')
    plt.ylabel('Proportion of nodes above threshold')
    plt.savefig(path+'/above_threshold_percentage.jpg', dpi=300)
    plt.close('all')


def plot_channel_profile(path, number_of_channels=1):

    from landlab.plot.channel_profile import analyze_channel_network_and_plot
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs
    all_dry = inputs.read_bool('all_dry')

    t = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        fr = FlowRouter(mg)
        pf = PitFiller(mg)
        mg = pf.pit_fill()
        mg = fr.route_flow(routing_flat=all_dry)
        plt.figure(1)
        profile_IDs, dists_upstr = analyze_channel_network_and_plot(mg, number_of_channels=number_of_channels)
        plt.xlabel('Distance upstream (m)')
        plt.ylabel('Elevation (m)')
        plt.xlim(0, 7000)
        plt.ylim(-50, 5)
        plt.title('t = '+str(int((i+1)*dt))+' years')
        plt.savefig(path + '/channel_profile_longest_t=' + str(t[i]) + '.jpg', dpi=300)
        plt.close('all')

        filepath = path + '/channel_nodes_longest_t=' + str(t[i]) + '.txt'
        output = file(filepath, 'w')
        output.write("%s\n" % len(profile_IDs))
        for profile_ID in range(len(profile_IDs)):
            output.write("%s\n" % len(profile_IDs[profile_ID]))
            for ele in profile_IDs[profile_ID]:
                output.write("%s\n" % ele)
            for ele in dists_upstr[profile_ID]:
                output.write("%s\n" % ele)
        output.close()

        print 'Finished loop', i+1


def plot_channel_profile_drainage(path):

    from landlab.plot.channel_profile import analyze_channel_network_and_plot
    from landlab.plot.channel_profile import plot_profiles
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs
    all_dry = inputs.read_bool('all_dry')

    t = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        fr = FlowRouter(mg)
        pf = PitFiller(mg)
        mg = pf.pit_fill()
        mg = fr.route_flow(routing_flat=all_dry)
        plt.figure(1)
        profile_IDs, dists_upstr = analyze_channel_network_and_plot(mg, number_of_channels=1)
        plt.close('all')
        plt.figure(1)
        plot_profiles(dists_upstr, profile_IDs, mg.at_node['drainage_area'])
        plt.xlabel('Distance upstream (m)')
        plt.ylabel('Drainage area (m*m)')
        plt.savefig(path + '/channel_profile_drainage_longest_t=' + str(t[i]) + '.jpg', dpi=300)
        plt.close('all')

        print 'Finished loop', i+1


def plot_channel(path, number_of_channels=1):

    from landlab.plot.channel_profile import analyze_channel_network_and_plot
    from landlab.plot.channel_profile import plot_profiles
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller
    from landlab.plot.imshow import imshow_node_grid

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs
    all_dry = inputs.read_bool('all_dry')

    t = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        fr = FlowRouter(mg)
        pf = PitFiller(mg)
        mg = pf.pit_fill()
        mg = fr.route_flow(routing_flat=all_dry)
        plt.figure(1)
        profile_IDs, dists_upstr = analyze_channel_network_and_plot(mg, number_of_channels=number_of_channels)
        plt.close('all')
        plt.figure(1)
        mg.at_node['channel_position'] = np.zeros(mg.number_of_nodes)
        for j in range(len(profile_IDs)):
            the_nodes = profile_IDs[j]
            mg.at_node['channel_position'][the_nodes] = 1
        im = imshow_node_grid(mg, 'channel_position', plot_name='t = '+str(int((i+1)*dt))+' years', \
                              grid_units = ['m','m'], cmap='Greys', allow_colorbar=False)
        plt.savefig(path + '/channel_longest_t=' + str(t[i]) + '.jpg', dpi=300)
        plt.close('all')

        print 'Finished loop', i+1


def plot_channel_unroute(path, number_of_channels=1):

    from landlab.plot.channel_profile import analyze_channel_network_and_plot
    from landlab.plot.channel_profile import plot_profiles
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller
    from landlab.plot.imshow import imshow_node_grid

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs
    all_dry = inputs.read_bool('all_dry')

    t = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        fr = FlowRouter(mg)
        pf = PitFiller(mg)
        mg = pf.pit_fill()
        mg = fr.route_flow(routing_flat=False)
        plt.figure(1)
        profile_IDs, dists_upstr = analyze_channel_network_and_plot(mg, number_of_channels=number_of_channels)
        plt.close('all')
        plt.figure(1)
        mg.at_node['channel_position_unroute'] = np.zeros(mg.number_of_nodes)
        for j in range(len(profile_IDs)):
            the_nodes = profile_IDs[j]
            mg.at_node['channel_position_unroute'][the_nodes] = 1
        im = imshow_node_grid(mg, 'channel_position_unroute', plot_name='t = '+str(int((i+1)*dt))+' years', \
                              grid_units = ['m','m'], cmap='Greys', allow_colorbar=False)
        plt.savefig(path + '/channel_unroute_longest_t=' + str(t[i]) + '.jpg', dpi=300)
        plt.close('all')

        print 'Finished loop', i+1


def plot_diffusion_vs_elevation(path):

    from landlab.components.diffusion.diffusion import LinearDiffuser

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs

    t = np.zeros(num_outs)
    percentage = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        original_topo = mg.at_node['topographic__elevation'].copy()
        lin_diffuse = LinearDiffuser(mg, input_file)
        mg = lin_diffuse.diffuse(dt)
        diffused_topo = mg.at_node['topographic__elevation'].copy()
        diffusion = diffused_topo-original_topo
        plt.close('all')
        plt.figure(1)
        plt.scatter(diffusion/100., mg.at_node['topographic__elevation'])
        plt.xlabel('Diffusion rate (m/yr)')
        plt.ylabel('Elevation (m)')
        plt.savefig(path+'/diff_elev_t='+str(t[i])+'.jpg', dpi = 300)
        plt.close('all')

        print 'Finished loop', i+1


def plot_drainage_network(path):
    from landlab.plot.imshow import imshow_node_grid
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs
    all_dry = inputs.read_bool('all_dry')

    initial_mg = build_model_grid(path, '/Topography_t=0.0.txt')
    min_elev = np.nanmin(initial_mg.at_node['topographic__elevation'][initial_mg.core_nodes])

    t = np.zeros(num_outs)
    channel_density = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        channels, = np.where(mg.at_node['topographic__elevation']<min_elev)
        channel_density[i] = len(channels)/float(mg.number_of_nodes)
        mg.at_node['channels'] = np.zeros(mg.number_of_nodes)
        mg.at_node['channels'][channels] = 1
        plt.close('all')
        plt.figure(1)
        im = imshow_node_grid(mg, 'channels', plot_name='t = '+str(int((i+1)*dt))+' years', \
                              grid_units = ['m','m'], cmap='Greys', allow_colorbar=False)
        plt.savefig(path + '/channels_t=' + str(t[i]) + '.jpg', dpi=300)
        plt.close('all')

        '''
        fr = FlowRouter(mg)
        pf = PitFiller(mg)
        if all_dry:
            mg = pf.pit_fill()
        mg = fr.route_flow(routing_flat=all_dry)
        channels = np.intersect1d(channels, mg.core_nodes)
        print np.nanmin(mg.at_node['drainage_area'][channels])
        '''
        print 'Finished loop', i+1

    save_result(t, channel_density, path+'/channel_density.txt')
    plt.close('all')
    plt.figure(1)
    plt.plot(t, channel_density)
    #plt.ylim([0,1])
    plt.xlabel('Time (years)')
    plt.ylabel('Proportion of channels')
    plt.savefig(path+'/channel_density.jpg', dpi=300)
    plt.close('all')


def analyze_sediment_volume(path):
    from landlab.plot.imshow import imshow_node_grid

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/num_outs
    all_dry = inputs.read_bool('all_dry')

    last_mg = build_model_grid(path, '/Topography_t=0.0.txt')

    t = np.zeros(num_outs)
    sediment_volume = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+0.5)*dt
        mg = build_model_grid(path, '/Topography_t='+str((i+1)*dt)+'.txt')
        elev_diff = last_mg.at_node['topographic__elevation']-mg.at_node['topographic__elevation']
        elev_diff = elev_diff[mg.core_nodes]
        sediment_volume[i] = np.sum(elev_diff)*mg.dx*mg.dx

        last_mg = mg
        print 'Finished loop', i+1

    save_result(t, sediment_volume, path+'/sediment_volume.txt')
    plt.close('all')
    plt.figure(1)
    sediment_volume = sediment_volume/1.e6
    plt.plot(t, sediment_volume)
    #plt.ylim([0,1])
    plt.xlabel('Time (years)')
    plt.ylabel('Volume of Sediments (*10^6 m^3)')
    plt.savefig(path+'/sediment_volume.jpg', dpi=300)
    plt.close('all')


def analyze_total_erosion_rate(path):
    from landlab.plot.imshow import imshow_node_grid

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/float(num_outs)
    all_dry = inputs.read_bool('all_dry')

    last_mg = build_model_grid(path, '/Topography_t=0.0.txt')

    t = np.zeros(num_outs)
    total_erosion_rate = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+0.5)*dt
        mg = build_model_grid(path, '/Topography_t='+str((i+1)*dt)+'.txt')
        elev_diff = last_mg.at_node['topographic__elevation']-mg.at_node['topographic__elevation']
        elev_diff = elev_diff[mg.core_nodes]
        total_erosion_rate[i] = np.sum(elev_diff)*mg.dx*mg.dx/dt

        last_mg = mg
        print 'Finished loop', i+1

    save_result(t, total_erosion_rate, path+'/total_erosion_rate.txt')
    plt.close('all')
    plt.figure(1)
    plt.plot(t, total_erosion_rate)
    #plt.ylim([0,1])
    plt.xlabel('Time (years)')
    plt.ylabel('Total erosion rate (m^3/year)')
    plt.savefig(path+'/total_erosion_rate.jpg', dpi=300)
    plt.close('all')


def analyze_diffusive_change(path):
    from landlab.components.diffusion.diffusion import LinearDiffuser

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/float(num_outs)
    all_dry = inputs.read_bool('all_dry')

    last_mg = build_model_grid(path, '/Topography_t=0.0.txt')

    t = np.zeros(num_outs)
    diffusion_rate_p = np.zeros(num_outs)
    diffusion_rate_n = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        original_topo = mg.at_node['topographic__elevation'].copy()
        lin_diffuse = LinearDiffuser(mg, input_file)
        mg = lin_diffuse.diffuse(dt=100.)
        diffused_topo = mg.at_node['topographic__elevation'].copy()
        change = diffused_topo-original_topo
        change = change*mg.dx*mg.dx
        change = change/100.
        diffusion_rate_p[i] = np.sum(change[np.where(change>0)])
        diffusion_rate_n[i] = np.sum(change[np.where(change<0)])

        mg = 1.
        lin_diffuse = 1.
        print 'Finished loop', i+1

    save_result(t, diffusion_rate_p, path+'/mass_change_by_diffusion_p.txt')
    save_result(t, diffusion_rate_n, path+'/mass_change_by_diffusion_n.txt')
    plt.close('all')
    plt.figure(1)
    plt.plot(t, diffusion_rate_p)
    plt.plot(t, diffusion_rate_n)
    #plt.ylim([0,1])
    plt.xlabel('Time (years)')
    plt.ylabel('Mass change by diffusion (m^3)')
    plt.savefig(path+'/mass_change_by_diffusion.jpg', dpi=300)
    plt.close('all')


def analyze_fluvial_change(path):
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.stream_power.fastscape_stream_power_JL import FastscapeEroder
    from landlab.components.sink_fill.pit_fill_pf import PitFiller
    from landlab.core.model_parameter_dictionary import MissingKeyError

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/float(num_outs)
    all_dry = inputs.read_bool('all_dry')
    k_sp = inputs.read_float('K_sp')
    try:
        threshold_AS = inputs.read_float('threshold_AS')
    except MissingKeyError:
        threshold_stream_power = inputs.read_float('threshold_stream_power')
        threshold_AS = threshold_stream_power/k_sp

    last_mg = build_model_grid(path, '/Topography_t=0.0.txt')

    t = np.zeros(num_outs)
    fluvial_erosion_rate = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        original_topo = mg.at_node['topographic__elevation'].copy()
        fr = FlowRouter(mg)
        pf = PitFiller(mg)
        sp = FastscapeEroder(mg, input_file, threshold_sp = threshold_AS*k_sp)
        filled_nodes = None
        if all_dry:
            mg = pf.pit_fill()
            filled_nodes, = np.where((mg.at_node['topographic__elevation']-original_topo)>0)
        mg = fr.route_flow(routing_flat=all_dry)
        old_topo = mg.at_node['topographic__elevation'].copy()
        mg = sp.erode(mg, dt=100., flooded_nodes=filled_nodes)
        new_topo = mg.at_node['topographic__elevation'].copy()
        change = new_topo-old_topo
        change = 0.-change*mg.dx*mg.dx
        change = change/100.
        fluvial_erosion_rate[i] = np.sum(change)

        mg = 1.
        fr = 1.
        pf = 1.
        sp = 1.
        print 'Finished loop', i+1

    save_result(t, fluvial_erosion_rate, path+'/mass_change_by_fluvial_erosion.txt')
    plt.close('all')
    plt.figure(1)
    plt.plot(t, fluvial_erosion_rate)
    #plt.ylim([0,1])
    plt.xlabel('Time (years)')
    plt.ylabel('Mass change by fluvial erosion (m^3)')
    plt.savefig(path+'/mass_change_by_fluvial_erosion.jpg', dpi=300)
    plt.close('all')

def analyze_fluvial_diffusive_change(path):
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.stream_power.fastscape_stream_power_JL import FastscapeEroder
    from landlab.components.sink_fill.pit_fill_pf import PitFiller
    from landlab.components.diffusion.diffusion import LinearDiffuser
    from landlab.core.model_parameter_dictionary import MissingKeyError

    print path

    input_file = path+'/coupled_params_sp.txt'
    inputs = ModelParameterDictionary(input_file)
    num_outs = inputs.read_int('number_of_outputs')
    runtime = inputs.read_float('total_time')
    dt = runtime/float(num_outs)
    all_dry = inputs.read_bool('all_dry')
    k_sp = inputs.read_float('K_sp')
    try:
        threshold_AS = inputs.read_float('threshold_AS')
    except MissingKeyError:
        threshold_stream_power = inputs.read_float('threshold_stream_power')
        threshold_AS = threshold_stream_power/k_sp

    last_mg = build_model_grid(path, '/Topography_t=0.0.txt')

    t = np.zeros(num_outs)
    fluvial_erosion_rate = np.zeros(num_outs)
    diffusion_rate_p = np.zeros(num_outs)
    diffusion_rate_n = np.zeros(num_outs)
    for i in range(num_outs):
        t[i] = (i+1)*dt
        mg = build_model_grid(path, '/Topography_t='+str(t[i])+'.txt')
        fr = FlowRouter(mg)
        pf = PitFiller(mg)
        sp = FastscapeEroder(mg, input_file, threshold_sp = threshold_AS*k_sp)
        lin_diffuse = LinearDiffuser(mg, input_file)
        change_f = np.zeros(10)
        change_d_p = np.zeros(10)
        change_d_n = np.zeros(10)
        for j in range(10):
            original_topo = mg.at_node['topographic__elevation'].copy()
            filled_nodes = None
            if all_dry:
                mg = pf.pit_fill()
                filled_nodes, = np.where((mg.at_node['topographic__elevation']-original_topo)>0)
            mg = fr.route_flow(routing_flat=all_dry)
            old_topo = mg.at_node['topographic__elevation'].copy()
            mg = sp.erode(mg, dt=dt/10., flooded_nodes=filled_nodes)
            new_topo = mg.at_node['topographic__elevation'].copy()
            change_f[j] = np.sum(old_topo-new_topo)*mg.dx*mg.dx
            mg.at_node['topographic__elevation'] = original_topo + new_topo - old_topo
            topo_before_diff = mg.at_node['topographic__elevation'].copy()
            mg = lin_diffuse.diffuse(dt=dt/10.)
            delta = mg.at_node['topographic__elevation']-topo_before_diff
            change_d_p[j] = np.sum(delta[np.where(delta>0)])*mg.dx*mg.dx
            change_d_n[j] = np.sum(delta[np.where(delta<0)])*mg.dx*mg.dx

        fluvial_erosion_rate[i] = np.sum(change_f)/dt
        diffusion_rate_p[i] = np.sum(change_d_p)/dt
        diffusion_rate_n[i] = np.sum(change_d_n)/dt

        mg = 1.
        fr = 1.
        pf = 1.
        sp = 1.
        lin_diffuse = 1.
        print 'Finished loop', i+1

    save_result(t, fluvial_erosion_rate, path+'/mass_change_by_fluvial_erosion.txt')
    plt.close('all')
    plt.figure(1)
    plt.plot(t, fluvial_erosion_rate)
    #plt.ylim([0,1])
    plt.xlabel('Time (years)')
    plt.ylabel('Mass change by fluvial erosion (m^3)')
    plt.savefig(path+'/mass_change_by_fluvial_erosion.jpg', dpi=300)
    plt.close('all')

    save_result(t, diffusion_rate_p, path+'/mass_change_by_diffusion_p.txt')
    save_result(t, diffusion_rate_n, path+'/mass_change_by_diffusion_n.txt')
    plt.close('all')
    plt.figure(1)
    plt.plot(t, diffusion_rate_p)
    plt.plot(t, diffusion_rate_n)
    #plt.ylim([0,1])
    plt.xlabel('Time (years)')
    plt.ylabel('Mass change by diffusion (m^3)')
    plt.savefig(path+'/mass_change_by_diffusion.jpg', dpi=300)
    plt.close('all')


def identify_channels(grid, base_nodes, elev_thr):
    from landlab.components.flow_routing.route_flow_dn_JL import FlowRouter
    from landlab.components.sink_fill.pit_fill_pf import PitFiller

    if not isinstance(base_nodes, (list, tuple, np.ndarray)):
        base_nodes = np.array([base_nodes])
    num_of_channels = len(base_nodes)
    n = grid.number_of_nodes
    fr = FlowRouter(grid)
    pf = PitFiller(grid)
    original_topo = grid.at_node['topographic__elevation'].copy()
    grid = pf.pit_fill()
    grid = fr.route_flow(routing_flat=True)
    grid.at_node['topographic__elevation'] = original_topo

    outlets = np.zeros(n)
    outlets[:] = -1.
    for node in base_nodes:
        outlets[node] = node
    node_stack = grid.at_node['flow__upstream_node_order']
    receiver = grid.at_node['flow__receiver_node']
    for i in range(n):
        outlets[node_stack[i]] = outlets[receiver[node_stack[i]]]

    drainage_area = grid.at_node['drainage_area']
    donor = np.arange(n)
    max_drainage = np.zeros(n)
    for i in range(n-1, -1, -1):
        node = node_stack[i]
        if drainage_area[node]>max_drainage[receiver[node]]:
            donor[receiver[node]] = node
            max_drainage[receiver[node]] = drainage_area[node]

    flow_to_boundary = identify_drained_area_elev_thr(grid, elev_thr)
    channel_list = []
    for outlet_node in base_nodes:
        channel = []
        channel.append(outlet_node)
        node = donor[outlet_node]
        if node == outlet_node:
            channel_list.append(channel)
            continue
        while flow_to_boundary[node]:
            channel.append(node)
            if node == donor[node]:
                break
            node = donor[node]
        channel_list.append(channel)

    channel_length = np.zeros(num_of_channels)
    for i in range(num_of_channels):
        distance_upstream = 0.
        node = base_nodes[i]
        for j in range(len(channel_list[i])-1):
            donor = channel_list[i][j+1]
            distance_upstream += np.sqrt((grid.node_x[node]-grid.node_x[donor])**2 \
                                          +(grid.node_y[node]-grid.node_y[donor])**2)
            node = donor
        channel_length[i] = distance_upstream

    channel_drainage = np.zeros(num_of_channels)
    for i in range(num_of_channels):
        drainage_of_channel, = np.where(np.logical_and(outlets==base_nodes[i], flow_to_boundary))
        channel_drainage[i] = len(drainage_of_channel)*grid.dx*grid.dx

    fr = 1
    pf = 1

    return (channel_list, channel_length, channel_drainage)
