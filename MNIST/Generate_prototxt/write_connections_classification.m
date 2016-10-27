function [] = write_connections_classification(fileID, flag_outgoing, flag_incoming, net_matrix, layer_index, depth_index, info_net)
% In this function we are going to write on the prototxt the connections from a node.
% CHANGE: In this case, we are going to change the functionality for the
% zipper network at the end. Now for the classification we will get a
% downward zipper.


if(flag_incoming.forward == 0 && flag_incoming.up == 0 && flag_incoming.down == 0 && flag_incoming.zipper_up == 0)
    error('we should have skipped this node')
    return
end

% First: we need to make this node from the incoming connections
make_node(fileID, flag_incoming, layer_index, depth_index, info_net);
fprintf(fileID, '\n');

% Second: We need to write the outgoing branches
if(~isequal(flag_outgoing,0))
    make_outgoing(fileID, flag_outgoing, layer_index, depth_index, info_net);
    fprintf(fileID, '\n \n');
end



end

function [] = make_outgoing(fileID, flag_outgoing, layer_index, depth_index, info_net)
% In this layer we are going to write the outgoing branches


depth_resolution_maps = info_net.depth_resolution_maps;

% This is at the end of the network
case_zipper_up = 0; 
if(isfield(flag_outgoing,'zipper_up'))
    case_zipper_up = flag_outgoing.zipper_up;
end

% This is at the start of the network
case_zipper_down = 0; 
if(isfield(flag_outgoing,'zipper_down'))
    case_zipper_down = flag_outgoing.zipper_down;
end

name_current_node = ['d',num2str(depth_index),'_',num2str(layer_index)];
name_current_node_up = ['d',num2str(depth_index),'_',num2str(layer_index),'_up'];
% name_current_node_down = ['d',num2str(depth_index),'_',num2str(layer_index),'_down'];

% This is for the case at the end/last layer
if(case_zipper_up)
   name_node_zipper_up = ['d',num2str(depth_index),'_',num2str(layer_index),'_up_zipper']; 
end

% This is for the case at the start layer
% if(case_zipper_down)
%    name_node_zipper_down = ['d',num2str(depth_index),'_',num2str(layer_index),'_down_zipper']; 
% end


if(flag_outgoing.up || flag_outgoing.down)
    fprintf(fileID,['# Branching units from ',name_current_node,' \n']);
end

if(flag_outgoing.up)
    % We have to branch out a connection by unpooling
    up_resolution = depth_resolution_maps(depth_index-1);
    fprintf(fileID,['# Unpooling --> ',num2str(up_resolution),' for depth ',num2str(depth_index-1),' \n']);
    string_up = ['layers {type:UNPOOLING  bottom:"',name_current_node,'"  top:"',name_current_node_up,'"  name:"',name_current_node_up,'"'];
    string_up = [string_up,' unpooling_param {unpool:MAX kernel_size:2 stride:2 unpool_size: ',num2str(up_resolution),'}} \n'];
    fprintf(fileID, string_up);
end

% if(flag_outgoing.down)
%     % We have to branch out a connection by downsampling
%     down_resolution = depth_resolution_maps(depth_index+1);
%     fprintf(fileID,['# Downsampling --> ',num2str(down_resolution),' for depth ',num2str(depth_index+1),' \n']);
%     string_down = ['layers {bottom:"',name_current_node,'"  top:"',name_current_node_down,'"  name:"',name_current_node_down,'" type:POOLING pooling_param {pool:TOPLEFT kernel_size:2 stride:2 }} \n'];
%     fprintf(fileID, string_down);
% end

if(flag_outgoing.down)
    fprintf(fileID,['# Downsample for ',name_current_node,', done automatically with convolution with stride 2 \n']);
end



% This for the case where we are the last layer
if(case_zipper_up)
    % We have to branch out a connection by unpooling for the zipper at end
    up_resolution = depth_resolution_maps(depth_index-1);
    fprintf(fileID,['# Zipper, Unpooling --> ',num2str(up_resolution),' for depth ',num2str(depth_index-1),' \n']);
    string_zipper_up = ['layers {type:UNPOOLING  bottom:"',name_current_node,'"  top:"',name_node_zipper_up,'"  name:"',name_node_zipper_up,'"'];
    string_zipper_up = [string_zipper_up,' unpooling_param {unpool:MAX kernel_size:2 stride:2 unpool_size: ',num2str(up_resolution),'}} \n'];
    fprintf(fileID, string_zipper_up);
end

% This is for the case where we are at the first layer
% if(case_zipper_down)
%     % We have to branch out a connection by downsampling
%     down_resolution = depth_resolution_maps(depth_index+1);
%     fprintf(fileID,['# Zipper, Downsampling --> ',num2str(down_resolution),' for depth ',num2str(depth_index+1),' \n']);
%     string_zipper_down = ['layers {bottom:"',name_current_node,'"  top:"',name_node_zipper_down,'"  name:"',name_node_zipper_down,'" type:POOLING pooling_param {pool:TOPLEFT kernel_size:2 stride:2 }} \n'];
%     fprintf(fileID, string_zipper_down);
% end

if(case_zipper_down)
    % We have to branch out a connection by downsampling
    fprintf(fileID,['# Zipper Downsample for ',name_current_node,', done automatically with convolution with stride 2 \n']);
end



end





function [] = make_node(fileID, flag_incoming, layer_index, depth_index, info_net)
debug = 0;
depth_channels = info_net.depth_channels;

channels_at_depth = depth_channels(depth_index);

name_current_node = ['d',num2str(depth_index),'_',num2str(layer_index)];
name_node_forward = ['d',num2str(depth_index),'_',num2str(layer_index-1)]; % Connection coming from the node at same level
name_node_up = ['d',num2str(depth_index-1),'_',num2str(layer_index-1),'_down']; % Connection coming from the node up      % This will be the name of the output
name_node_down = ['d',num2str(depth_index+1),'_',num2str(layer_index-1),'_up']; % Connection coming from the node below

% We have skipped the intermediate step
name_node_up_incoming = ['d',num2str(depth_index-1),'_',num2str(layer_index-1)]; % Connection coming from the node up

% This is at the end of the network
case_zipper_down = 0; 
if(isfield(flag_incoming,'zipper_down'))
    case_zipper_down = 1;
end

% This is at the start of the network
case_zipper_up = 0; 
if(isfield(flag_incoming,'zipper_up'))
    case_zipper_up = 1;
end

% Only if down zipper is there
if(case_zipper_down)
    name_node_zipper_down = ['d',num2str(depth_index+1),'_',num2str(layer_index),'_up_zipper']; % Connection coming from the node below, at same layer
end

% Only if up zipper is there
if(case_zipper_up)
    name_node_zipper_up = ['d',num2str(depth_index-1),'_',num2str(layer_index),'_down_zipper']; % Connection coming from the node above, at same layer
    % Now we have not created this intermediate zipper.
    name_node_zipper_up_incoming = ['d',num2str(depth_index-1),'_',num2str(layer_index)]; % Connection coming from the node above, at same layer
end


fprintf(fileID,['# Depth ',num2str(depth_index),': Forming ',name_current_node,'\n']);
% fprintf(fileID,['# Forming ',name_current_node,' \n']);

number_padding_MNIST = 1;
% if(depth_index == 1 && layer_index == 1)  % This was the case where we used image list and already feeded the jittered image.
%     number_padding_MNIST = 3;
% end

% Forming the data connections
% Forward
if(flag_incoming.forward )
    fprintf(fileID,'# Convolve Data at same level \n');
    string_forward = ['layers {bottom: "',name_node_forward,'"  top: "',name_node_forward,'_conv"  name: "',name_node_forward,'_conv"  type:CONVOLUTION blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0 '];
    string_forward = [string_forward,'\nconvolution_param {num_output: ',num2str(channels_at_depth),' pad: ',num2str(number_padding_MNIST),' kernel_size: 3 weight_filler {type: "gaussian" std: 0.01 } bias_filler{type: "constant" value: 0}  engine:CAFFE group:1}} \n'];
    fprintf(fileID,string_forward);
end

% Put the clause when this is the first incoming node.
% We can just name the data to be node d0_0


% Down
if(flag_incoming.down)
    fprintf(fileID,'# Convolve over data from down \n');
    string_down = ['layers {bottom: "',name_node_down,'"  top: "',name_node_down,'_conv"  name: "',name_node_down,'_conv"  type:CONVOLUTION blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0 '];
    string_down = [string_down ,'\nconvolution_param {num_output: ',num2str(channels_at_depth),' pad: 1 kernel_size: 3 weight_filler {type: "gaussian" std: 0.01 } bias_filler{type: "constant" value: 0}  engine:CAFFE group:1}} \n'];
    fprintf(fileID,string_down);
end


% Up
if(flag_incoming.up)
    fprintf(fileID,'# Convolve over data from up \n');
    string_up = ['layers {bottom: "',name_node_up_incoming,'"  top: "',name_node_up,'_conv"  name: "',name_node_up,'_conv"  type:CONVOLUTION blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0 '];
    %string_up = [string_up,'\nconvolution_param {num_output: ',num2str(channels_at_depth),' pad: 1 kernel_size: 3 weight_filler {type: "gaussian" std: 0.01 } bias_filler{type: "constant" value: 0}  engine:CAFFE group:1}} \n'];
    % We will do a stride 2, so as to downsample
    string_up = [string_up,'\nconvolution_param {num_output: ',num2str(channels_at_depth),' stride: 2 pad: 1 kernel_size: 3 weight_filler {type: "gaussian" std: 0.01 } bias_filler{type: "constant" value: 0}  engine:CAFFE group:1}} \n'];
    fprintf(fileID,string_up);
end

% Zipper at end
if(case_zipper_down)
    fprintf(fileID,'# Convolve over data from zipper from below\n');
    string_zip_down = ['layers {bottom: "',name_node_zipper_down,'"  top: "',name_node_zipper_down,'_conv"  name: "',name_node_zipper_down,'_conv"  type:CONVOLUTION blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0 '];
    string_zip_down = [string_zip_down,'\nconvolution_param {num_output: ',num2str(channels_at_depth),' pad: 1 kernel_size: 3 weight_filler {type: "gaussian" std: 0.01 } bias_filler{type: "constant" value: 0}  engine:CAFFE group:1}} \n'];
    fprintf(fileID,string_zip_down);
end

% Zipper at start
if(case_zipper_up)
    fprintf(fileID,'# Convolve over data from zipper coming from top \n');
    string_zip_up = ['layers {bottom: "',name_node_zipper_up_incoming,'"  top: "',name_node_zipper_up,'_conv"  name: "',name_node_zipper_up,'_conv"  type:CONVOLUTION blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0 '];
    %string_zip_up = [string_zip_up,'\nconvolution_param {num_output: ',num2str(channels_at_depth),' pad: 1 kernel_size: 3 weight_filler {type: "gaussian" std: 0.01 } bias_filler{type: "constant" value: 0}  engine:CAFFE group:1}} \n'];
    % We will do a stride 2, so as to downsample
    string_zip_up = [string_zip_up,'\nconvolution_param {num_output: ',num2str(channels_at_depth),' stride: 2 pad: 1 kernel_size: 3 weight_filler {type: "gaussian" std: 0.01 } bias_filler{type: "constant" value: 0}  engine:CAFFE group:1}} \n'];
    fprintf(fileID,string_zip_up);
end



% Now we can sum up the data-connections
fprintf(fileID,'\n # Perform the sum, BN and Relu \n');

% We will grow the string for the sum
% Zipper case should be handled here on its own
fields = fieldnames(flag_incoming);
number_bottom_blobs = 0;
string_bottom = [];
for i = 1:numel(fields)
    temp_flag = flag_incoming.(fields{i});
    if(debug)
        fprintf('flag_incoming.%s has a value %d \n',fields{i},temp_flag);
    end
    
    if(temp_flag) % If this connection is there
        temp_string = ['bottom:"',eval(strcat('name_node_',fields{i})),'_conv" '];
        string_bottom = horzcat(string_bottom, temp_string);
        number_bottom_blobs = number_bottom_blobs + 1;
    end
end

% If we just have 1 bottom blob as input we will use power layer, which by default does identity transform
assert(number_bottom_blobs > 0);
if(number_bottom_blobs == 1)
    layer_used = 'POWER';
else
    layer_used = 'ELTWISE';
end

assert(~isempty(string_bottom),'Atleast 1 bottom should be fed')
string_sum = ['layers{ ',string_bottom,' top:"',name_current_node,'" name:"',name_current_node,'"type:',layer_used,' } \n'];
    
string_BN = ['layers { bottom: "',name_current_node,'" top: "',name_current_node,'" name: "bn',name_current_node,'" type: BN bn_param {scale_filler {type: "constant" value: 1} shift_filler {type: "constant" value: 0.001 }}} \n'];
string_relu = ['layers { bottom: "',name_current_node,'" top: "',name_current_node,'" name: "relu',name_current_node,'" type: RELU } \n'];

fprintf(fileID,string_sum);
fprintf(fileID,string_BN);
fprintf(fileID,string_relu);

end




