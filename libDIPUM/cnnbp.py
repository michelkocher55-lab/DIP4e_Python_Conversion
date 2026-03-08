import numpy as np
from scipy import signal

def expand(A, S):
    """
    Replicate and tile each element of an array A by vector S.
    A: numpy array
    S: list/tuple of scaling factors per dimension.
    """
    for axis, scale in enumerate(S):
        if scale > 1:
            A = np.repeat(A, scale, axis=axis)
    return A

def flipall(X):
    """Flip all dimensions."""
    sl = [slice(None, None, -1)] * X.ndim
    return X[tuple(sl)]

def rot180(X):
    """Rotate 180 degrees (flip dim 0 and 1). Assuming 2D."""
    return np.rot90(X, 2)

def cnnbp(net, y):
    """
    Backpropagation for CNN.
    
    net = cnnbp(net, y)
    
    Parameters
    ----------
    net : dict
        CNN network structure.
    y : numpy.ndarray
        Target values (labels).
        
    Returns
    -------
    net : dict
        Network with computed gradients 'd', 'dk', 'db', etc.
    """
    
    n = len(net['layers']) # Number of layers (including input which is 0-indexed in Python?)
    # Adjust indexing: MATLAB 1-based. Python 0-based.
    # MATLAB: layers 1..n. 
    # Python: layers 0..n-1.
    # net.layers indexing shifted by -1 compared to MATLAB.
    # MATLAB inputs net.layers{n} is output layer? No, fully connected is outside layers?
    # Actually, MATLAB code iterates layers l=...
    # The structure suggests `net.layers` stores Conv/Sub layers. 
    # FF part `ffW` is separate.
    # `net.o` seems to be the final output of the FF part.
    
    # 1. Error
    net['e'] = net['o'] - y
    
    # 2. Loss
    # net.L = 1/2 * sum(net.e(:).^2) / size(net.e, 2)
    # size(e, 2) implies batch size is dim 1 (columns)?
    # MATLAB: e is (OutputDim x BatchSize).
    batch_size = net['e'].shape[1]
    net['L'] = 0.5 * np.sum(net['e']**2) / batch_size
    
    # 3. Backprop Deltas
    # Output Delta
    # net.od = net.e .* (net.o .* (1 - net.o))
    net['od'] = net['e'] * (net['o'] * (1 - net['o']))
    
    # Feature Vector Delta
    # net.fvd = (net.ffW' * net.od)
    # If net.o is (OutDim x Batch), ffW is (OutDim x Features)?
    # MATLAB: ffW * fv -> o. So ffW is (OutDim x Features).
    # Then ffW' * od -> (Features x OutDim) * (OutDim x Batch) -> (Features x Batch).
    net['fvd'] = np.dot(net['ffW'].T, net['od'])
    
    # Apply Sigmoid derivative if last internal layer is 'c'
    # MATLAB: if strcmp(net.layers{n}.type, 'c')
    # Use net['layers'][-1]
    last_layer = net['layers'][-1]
    if last_layer['type'] == 'c':
        net['fvd'] = net['fvd'] * (net['fv'] * (1 - net['fv']))
        
    # Reshape feature vector deltas into output map style
    # sa = size(net.layers{n}.a{1})
    # fvnum = sa(1) * sa(2)
    # In Python, 'a' likely list of (H, W, Batch)
    
    # Use output map size from first map
    sa = last_layer['a'][0].shape
    # sa is (H, W, Batch)
    H, W = sa[0], sa[1]
    fvnum = H * W
    
    # Initialize 'd' cell for last layer
    # MATLAB: net.layers{n}.d{j} = ...
    last_layer['d'] = [None] * len(last_layer['a'])
    
    for j in range(len(last_layer['a'])):
        # reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3))
        # slice fvd rows.
        # Python: j*fvnum : (j+1)*fvnum
        start_idx = j * fvnum
        end_idx = (j + 1) * fvnum
        
        chunk = net['fvd'][start_idx:end_idx, :] # (H*W, Batch)
        # Reshape to (H, W, Batch).
        # Need to be careful with 'F' vs 'C' ordering if verifying against MATLAB.
        # MATLAB reshape fills columns first.
        # If flatten was done C-order, reshape C-order.
        # MATLAB `fv = [a{1}(:); a{2}(:); ...]`. `(:)` is column-major.
        # So we should probably treat as Fortran order if we want exact match, 
        # or just ensure consistency between fv generation (cnnff) and this.
        # Assuming standard Python usage: C-order.
        
        last_layer['d'][j] = chunk.reshape(H, W, batch_size) # Default C-order?
        # If we care about exact numerical match with MATLAB transcoding we might need order='F'.
        # I'll stick to default for now unless testing fails.
        
    # Backprop Loop
    # MATLAB: for l = (n - 1) : -1 : 1
    # Python: range(n-2, -1, -1) -> layers indexes (n-2) down to 0 input?
    # No, layer 0 is input? Usually backprop stops at layer 1 (first hidden).
    # MATLAB loop goes down to 1. IF layer 1 is input, it might error if we access l.d? 
    # Usually layer 1 in MATLAB cnn is 'i' (input)? Or 'c'?
    # If l=1 is 'c', we compute d{1}.
    
    num_layers = len(net['layers'])
    # Indices 0 to num_layers-1.
    # Last layer handled above (index num_layers-1).
    # So loop num_layers-2 down to 0.
    
    for l in range(num_layers - 2, -1, -1):
        layer = net['layers'][l]
        next_layer = net['layers'][l+1]
        
        if layer['type'] == 'c':
            # Next layer must be 's'
            layer['d'] = [None] * len(layer['a'])
            scale = next_layer['scale']
            
            for j in range(len(layer['a'])):
                # d{j} = a{j} * (1-a) * (expand(next.d{j}) / scale^2)
                
                # expand next_layer['d'][j]
                # next_layer['d'][j] is (H_next, W_next, Batch).
                # expand dims 0 and 1. Dim 2 (Batch) scale 1.
                upsampled = expand(next_layer['d'][j], [scale, scale, 1])
                
                term = upsampled / (scale**2)
                layer['d'][j] = layer['a'][j] * (1 - layer['a'][j]) * term
                
        elif layer['type'] == 's':
            # Next layer must be 'c'
            layer['d'] = [None] * len(layer['a'])
            
            for i in range(len(layer['a'])):
                # z = zeros
                z = np.zeros_like(layer['a'][0])
                
                for j in range(len(next_layer['a'])):
                    # z = z + convn(next.d{j}, rot180(next.k{i}{j}), 'full')
                    
                    next_d = next_layer['d'][j]
                    kernel = next_layer['k'][i][j]
                    rot_kernel = rot180(kernel)
                    
                    # Convolve 3D d with 2D kernel.
                    # scipy convolve with mode 'full'.
                    # Broadcast kernel to (Kh, Kw, 1)
                    res = signal.convolve(next_d, rot_kernel[..., np.newaxis], mode='full')
                    z = z + res
                    
                layer['d'][i] = z
                
    # Calc Gradients
    # MATLAB: for l = 2 : n
    # Python: range(1, num_layers)
    
    for l in range(1, num_layers):
        layer = net['layers'][l]
        prev_layer = net['layers'][l-1]
        
        if layer['type'] == 'c':
            layer['dk'] = []
            layer['db'] = []
            
            # Init empty grid for dk: [i][j]
            # Wait, MATLAB: net.layers{l}.dk{i}{j}.
            # Python Structure: layer['dk'] should be list of lists?
            # Or dict? Let's use list of lists to match 'k'.
            
            # Need to initialize layer['dk'] structure compatible with k
            # layer['k'] is [i][j] (input i -> output j)
            num_inputs = len(prev_layer['a'])
            num_outputs = len(layer['a'])
            
            layer['dk'] = [[None for _ in range(num_outputs)] for _ in range(num_inputs)]
            layer['db'] = [None] * num_outputs
            
            for j in range(num_outputs):
                for i in range(num_inputs):
                    # convn(flipall(prev.a{i}), layer.d{j}, 'valid') / batch_size
                    
                    prev_a_flipped = flipall(prev_layer['a'][i]) # (H, W, B) -> flipped all
                    # Wait, flipall includes batch dim?
                    # MATLAB flipall flips ALL dims.
                    # prev.a{i} is (H, W, B). Flipped: (-H, -W, -B).
                    # layer.d{j} is (outH, outW, B).
                    
                    # Gradient of kernel involves correlation (convolution with flipped).
                    # The dimensions must effectively sum out the batch dimension?
                    # If we convolve (H, W, B) with (outH, outW, B) 'valid', 
                    # do we get (Kh, Kw, 1)?
                    # scipy convolve dim behavior:
                    # In1 shape (H, W, B), In2 shape (outH, outW, B).
                    # Result shape (H - outH + 1, W - outW + 1, 1).
                    # Yes! This sums over B if 'valid' reduces it.
                    # B - B + 1 = 1.
                    
                    dk_res = signal.convolve(prev_a_flipped, layer['d'][j], mode='valid')
                    # Reshape to 2D
                    dk_res = np.squeeze(dk_res)
                    
                    layer['dk'][i][j] = dk_res / batch_size
                    
                # db
                # sum(layer.d{j}(:)) / batch_size
                layer['db'][j] = np.sum(layer['d'][j]) / batch_size # Scalar
                
    # dffW, dffb
    # net.dffW = net.od * (net.fv)' / batch_size
    # od: (OutDim, B). fv: (Feat, B).
    # od * fv.T -> (OutDim, Feat).
    net['dffW'] = np.dot(net['od'], net['fv'].T) / batch_size
    
    # net.dffb = mean(net.od, 2)
    # Mean across batch (axis 1)
    net['dffb'] = np.mean(net['od'], axis=1, keepdims=True)
    
    return net
