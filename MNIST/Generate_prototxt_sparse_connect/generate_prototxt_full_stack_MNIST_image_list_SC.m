function [ ] = generate_prototxt_full_stack_MNIST_image_list_SC(number_layers, number_channels)

% This code is copied from generate_prototxt_full_stack_MNIST_image_list
% and we are going to modify it to have the sparse connect, with the
% efficient implementation we had in BLAS.

% This is done only for the final result in the paper:

% Changes:
% For convolution layers, we will introduce sparse_connect:true!
% We will change the input data.
% We have copied write_solver_file from the Generate_prototxt_sparse_connect
% We have to manually delete the flag sparse_connect: true for the case of d1_1

info_depth.d1.valid_index = 1:number_layers;
info_depth.d2.valid_index = 1:number_layers;
info_depth.d3.valid_index = 1:number_layers;
info_depth.d4.valid_index = 1:number_layers;
info_depth.d5.valid_index = 1:number_layers;
info_depth.d6.valid_index = 1:number_layers;

is_CPU = 1;


% We will write the solver prototxt as well
write_solver_file(number_layers, number_channels, is_CPU);



% Open a file
fileID = fopen(['6x',num2str(number_layers),'_',num2str(number_channels),'channels_sparse_connect_paper.prototxt'],'w');

% Error checks
info_net = error_check(info_depth);



% Convert net description to a matrix form
net_matrix = convert_net_into_matrix(info_net, info_depth);

info_net.depth_resolution_maps = [32,16,8,4,2,1];
info_net.depth_channels = repmat(number_channels,size(info_net.depth_resolution_maps));

% Write the standard input output ( We can copy and paste this later)
% write_start(fileID, info_net);
% write_MNIST_start(fileID, info_net)
write_MNIST_start_image_list(fileID, info_net)

% Function to write the data input:
% Do by hand:

for i = 1:info_net.layers
   % Step one by one on layer, except the last, since we have to zip it up
   
    for j = 1:info_net.depth
        
        % We deal with this node only if it is a valid point
        if(net_matrix(j,i))
            % Check the outgoing connections to branch
            flag_outgoing = check_outgoing_at_this_node(i,j,net_matrix);
            
            % Check the incoming connections to connect
            flag_incoming = check_incoming_at_this_node(i, j, net_matrix);

            % Write the connections 
            write_connections_classification_SC(fileID, flag_outgoing, flag_incoming, net_matrix, i, j, info_net);
        end
        
    end
    
end

% Zipper network: Now we will traverse in the same order as before for the classification network.

% Write the end code for the objective function ( Has to be modified by hand)
write_end(fileID, info_net.layers, info_net.depth);
fclose(fileID);


end


function [] = write_solver_file(number_layers, number_channels, is_CPU)

if(is_CPU)
    fileID = fopen(['CPU_solver_6x',num2str(number_layers),'_',num2str(number_channels),'channels_sparse_connect_paper.prototxt'],'w');
else
    fileID = fopen(['GPU_solver_6x',num2str(number_layers),'_',num2str(number_channels),'channels_sparse_connect_paper.prototxt'],'w');    
end

my_print(fileID,'# The train/test net protocol buffer definition') 
% This is different now
my_print(fileID,['net: "./6x',num2str(number_layers),'_',num2str(number_channels),'channels_sparse_connect_paper.prototxt','"'])
my_print(fileID,'test_iter: 1000')
my_print(fileID,'test_interval: 20000') % We will just test at start and end of max_iter
my_print(fileID,'')

my_print(fileID,'# The base learning rate, momentum and the weight decay of the network.')
my_print(fileID,'base_lr: 0.01')
my_print(fileID,'momentum: 0.9')
my_print(fileID,'weight_decay: 0.0005')
my_print(fileID,'iter_size: 8')
my_print(fileID,'')

my_print(fileID,'# The learning rate policy')
my_print(fileID,'lr_policy: "step"')
my_print(fileID,'stepsize: 5000')
my_print(fileID,'gamma: 0.1')
my_print(fileID,'')


my_print(fileID,'# Display every 1000 iterations')
my_print(fileID,'display: 1000')
my_print(fileID,'max_iter: 20000')
my_print(fileID,'snapshot: 2000')
my_print(fileID,['snapshot_prefix: "./6x',num2str(number_layers),'_',num2str(number_channels),'channels_sparse_connect_paper_LR_0.01_"'])

if(is_CPU)
my_print(fileID,'solver_mode: CPU')
else
my_print(fileID,'solver_mode: GPU')
end
my_print(fileID,'random_seed :1 ')

fclose(fileID);
end

function flag_incoming = check_incoming_at_this_node(layer_index, depth_index, net_matrix)
% We will check for the three incoming connections


% Checking for forward connect
% This will occur only for the first depth as rest would be feeded by input zipper
    if( layer_index == 1 )                                                 % CHANGE
        if(depth_index == 1)
            % If we are at first layer
            flag_incoming.forward = 1;
        else
            flag_incoming.forward = 0; 
        end
    else
        % If there was a node behind at same level
        flag_incoming.forward = net_matrix(depth_index,layer_index-1); 
    end

    
% Checking for upward connect
    
    % If we are already at the first depth, or at the first layer
    if( depth_index == 1 || layer_index == 1)
        flag_incoming.up = 0;
    else
        flag_incoming.up = net_matrix(depth_index-1,layer_index-1);
    end

% Checking for downward connect
    
    % If we are the last depth, or at first layer
    if(depth_index == size(net_matrix,1) || layer_index == 1)
         flag_incoming.down = 0;
    else
        flag_incoming.down = net_matrix(depth_index+1,layer_index-1);
    end
    
% Zipper network
% % If we are at the last layer, and not the last depth then we get a zipper.
%     if(depth_index ~= size(net_matrix,1) && layer_index == size(net_matrix,2))
%         flag_incoming.zipper_down = 1; 
%     end

% Now we will get a zipper from up, for the classification network
% If we are at the last layer, and not at the first depth then we will get a zipper_up
    if(depth_index ~= 1 && layer_index == size(net_matrix,2))
        flag_incoming.zipper_up = 1; 
    end
    
    
% If we are at the first layer, and not at the first depth then we get zipper_up
    if(depth_index ~= 1 && layer_index == 1)
        flag_incoming.zipper_up = 1; 
    end


end


function flag_outgoing = check_outgoing_at_this_node(layer_index, depth_index, net_matrix)
%  We have to check if we would have to branch up or below

% We will enter this loop for the node, only if it is in the valid index
if(net_matrix(depth_index,layer_index) == 0)
    flag_outgoing =0;
    return
end

% This will happen only for the last layer and at first depth
% if(layer_index == size(net_matrix,2) && depth_index == 1)
%     flag_outgoing = 0;
%     return 
% end

% Now there is a difference since we do not finish at last layer and depth
% 1, rather at last layer and last depth
if(layer_index == size(net_matrix,2) && depth_index == size(net_matrix,1))
    flag_outgoing = 0;
    return 
end


% ZIPPER network 
% % For last layer:
% % If we are still at last layer, we will just fork out an upsample 
% if(layer_index == size(net_matrix,2) && depth_index > 1)
%     flag_outgoing.up = 0;
%     flag_outgoing.down = 0;
%     flag_outgoing.zipper_up = 1;   
%     return 
% end

% Now we will go down zipper for the classification case
% If we are still at last layer, we will just fork out an upsample 
if(layer_index == size(net_matrix,2) && depth_index ~= size(net_matrix,1))
    flag_outgoing.up = 0;
    flag_outgoing.down = 0;
    flag_outgoing.zipper_down = 1;   
    return 
end



% For first layer:
% If we are at the first layer, we will fork out a downsample
if(layer_index == 1 && depth_index ~= size(net_matrix,1))
    flag_outgoing.up = 0;
    flag_outgoing.down = 1;
    flag_outgoing.zipper_down = 1;   
    
    if(depth_index > 1)
        flag_outgoing.up = 1;
    end
    
    return 
end


% Checking for up connect
    
    % If we are at the top layer
    if(depth_index == 1 )
        flag_outgoing.up = 0;
    else
    % If there is no valid index at the layer above    
        flag_outgoing.up = net_matrix(depth_index-1,layer_index+1);
    end

% Checking for down connect

    % If we are at the last depth
    if(depth_index == size(net_matrix,1) )
        flag_outgoing.down = 0;
    else
    % If there is no valid index at the depth below
        flag_outgoing.down = net_matrix(depth_index + 1, layer_index + 1);
    end
end





function [info_net] = error_check(info_depth)

fields = fieldnames(info_depth);
net_depth = numel(fields);

temp_indices = [];
for i = 1:numel(fields)
    temp_indices = horzcat(temp_indices,info_depth.(fields{i}).valid_index);
end

net_layers = numel(unique(temp_indices));

% The last layer index should be equal to the length of the net
assert(net_layers == max(temp_indices));

info_net.depth = net_depth;
info_net.layers = net_layers;
end

function [] = write_end(fileID, last_layer_index, last_depth)

% Change
name_last_node = ['d',num2str(last_depth),'_',num2str(last_layer_index)];

fprintf(fileID,['layers { name: "class-score" type: CONVOLUTION   bottom: "',name_last_node,'" top: "class-score"  blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0  \nconvolution_param { num_output: 10 kernel_size: 1 weight_filler {type: "gaussian" std: 0.01 } bias_filler{type: "constant" value: 0}}} \n\n']);
fprintf(fileID,'layers {  name: "softmax-loss"  type: SOFTMAX_LOSS  bottom: "class-score"  bottom: "class-label"  top: "softmax-loss"} \n');
fprintf(fileID,'layers {  name: "accuracy"  type: ELTWISE_ACCURACY  bottom: "class-score"  bottom: "class-label"  top: "accuracy" top: "weighted-accuracy" include: { phase: TEST }}\n');
end

function [] = write_start(fileID, info_net)
fprintf(fileID,'#hello \n');
fprintf(fileID,'name: "dummy" \n');
org_depth = info_net.depth_resolution_maps(1);


for i = 1
    fprintf(fileID,['layers{ name: "d',num2str(i),'_0" type: DUMMY_DATA \ndummy_data_param { num: 1 channels: 3 height: ',num2str(org_depth/2^(i-1)),' width: ',num2str(org_depth/2^(i-1)),'  data_filler { type: "uniform"   min: 10  max: 256}} top: "d',num2str(i),'_0"} \n']);
end

fprintf(fileID,'layers{ name: "dummy0" type: DUMMY_DATA \ndummy_data_param { num: 1 channels: 1 height: 1 width: 1 data_filler { type: "uniform"   min: 1   max: 3}} top: "class-label" } \n \n \n');



end
% 
% function [] = write_MNIST_start(fileID, info_net)
% fprintf(fileID,'#hello \n');
% fprintf(fileID,'name: "dummy" \n');
% 
% my_print(fileID,'layers {')
% my_print(fileID,'name: "mnist"  type: DATA  top: "d1_0"  top: "class-label"')
% my_print(fileID,'data_param {     source: "/scratch/gpuhost7/ssaxena/software/caffemodified/caffe_own/examples/mnist/mnist_train_lmdb"    backend: LMDB    batch_size: 64  }')
% my_print(fileID,'transform_param {    scale: 0.00390625  }')
% my_print(fileID,'include: { phase: TRAIN }')
% my_print(fileID,'}')
% 
% my_print(fileID, '\n')
% 
% my_print(fileID,'layers {')
% my_print(fileID,'name: "mnist"  type: DATA  top: "d1_0"  top: "class-label"')
% my_print(fileID,'data_param {     source: "/scratch/gpuhost7/ssaxena/software/caffemodified/caffe_own/examples/mnist/mnist_test_lmdb"    backend: LMDB    batch_size: 100  }')
% my_print(fileID,'transform_param {    scale: 0.00390625  }')
% my_print(fileID,'include: { phase: TEST }')
% my_print(fileID,'}')
% 
% 
% end


function [] = write_MNIST_start_image_list(fileID, info_net)
fprintf(fileID,'#hello \n');
fprintf(fileID,'name: "dummy" \n');

my_print(fileID,'layers {')
my_print(fileID,'name: "mnist"  type: IMAGE_DATA top: "d1_0"  top: "class-label"')
my_print(fileID,'image_data_param {   source: "/home/lear/ssaxena/Desktop/Backup/New_experiments_for_ECCV16/MNIST/generate_data_augmentation/Read_own_MNIST/train_image_list.txt"   ')
my_print(fileID,'batch_size: 8 is_color: 0 }')
my_print(fileID,'transform_param {    scale: 0.00390625  }')
my_print(fileID,'include: { phase: TRAIN }')
my_print(fileID,'}')

my_print(fileID, '\n')

my_print(fileID,'layers {')
my_print(fileID,'name: "mnist"  type: IMAGE_DATA top: "d1_0"  top: "class-label"')
my_print(fileID,'image_data_param {   source: "/home/lear/ssaxena/Desktop/Backup/New_experiments_for_ECCV16/MNIST/generate_data_augmentation/Read_own_MNIST/test_image_list.txt"   ')
my_print(fileID,'batch_size: 10 is_color: 0 }')
my_print(fileID,'transform_param {    scale: 0.00390625  }')
my_print(fileID,'include: { phase: TEST }')
my_print(fileID,'}')


end


function my_print(fileID, string_temp)

fprintf(fileID,string_temp);
fprintf(fileID,'\n');

end


function net_matrix = convert_net_into_matrix(info_net, info_depth)

fields = fieldnames(info_depth);
net_matrix = zeros(info_net.depth,info_net.layers);

for depth = 1:info_net.depth
    temp_indices = info_depth.(fields{depth}).valid_index;
    net_matrix(depth,temp_indices) = 1;
end

end
