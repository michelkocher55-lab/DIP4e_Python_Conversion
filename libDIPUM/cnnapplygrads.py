import numpy as np

def cnnapplygrads(net, opts):
    """
    Updates the weights and biases of the CNN using gradients.
    
    net = cnnapplygrads(net, opts)
    
    Parameters
    ----------
    net : dict
        CNN network structure.
        Expected keys:
            'layers': list of layer dicts.
            'ffW': Fully connected weights (numpy array).
            'ffb': Fully connected biases (numpy array).
            'dffW': Gradient for ffW.
            'dffb': Gradient for ffb.
        Layer dict structure (for type 'c'):
            'type': 'c'
            'k': list of lists of 2D numpy arrays (kernels). k[input_map][output_map]
            'dk': Gradients for k.
            'b': list of 1D numpy arrays/scalars (biases). b[output_map]
            'db': Gradients for b.
            'a': list of output maps (used for count).
    opts : dict
        Options. Expected key: 'alpha' (learning rate).
        
    Returns
    -------
    net : dict
        Updated network.
    """
    
    alpha = opts['alpha']
    
    # Iterate layers. 
    # MATLAB: for l = 2 : numel(net.layers)
    # Python: range(1, len(net['layers']))
    # Assuming layer 0 is input.
    
    for l in range(1, len(net['layers'])):
        layer = net['layers'][l]
        prev_layer = net['layers'][l-1]
        
        if layer['type'] == 'c':
            # Number of output maps
            # MATLAB: numel(net.layers{l}.a)
            # We assume 'a' exists. Or we can use len(layer['k'][0]) if 'a' is just activation storage.
            # The MATLAB code loops j=1:numel(a).
            # Let's assume 'a' is present (even if empty list references, simpler to mock length).
            # Or assume 'outputmaps' count is stored.
            # Using 'a' from struct.
            
            num_output_maps = len(layer['a'])
            num_input_maps = len(prev_layer['a'])
            
            for j in range(num_output_maps):
                for ii in range(num_input_maps):
                    # net.layers{l}.k{ii}{j} update
                    # In python list of lists: layer['k'][ii][j]
                    layer['k'][ii][j] = layer['k'][ii][j] - alpha * layer['dk'][ii][j]
                
                # Biases
                layer['b'][j] = layer['b'][j] - alpha * layer['db'][j]
                
    # Update Fully Connected parameters
    net['ffW'] = net['ffW'] - alpha * net['dffW']
    net['ffb'] = net['ffb'] - alpha * net['dffb']
    
    return net
