addpath(genpath(['MinMaxSelection']));
addpath(genpath(['utils']));
addpath(genpath(['metrics']));


%% create neighbor (label: 1) and non-neighbor (label: -1)graphs from pairs of objects
data_names = {'animal'};
split_ratios = [0.7];
num_sample   = 10;

for dataset = data_names
    disp(strcat('--- dataset = ', dataset , '-----'))
    for split_ratio = split_ratios
        disp(strcat('--- split-ratio = ', num2str(split_ratio) , '-----'))
        dataPath = strcat('data/', dataset, '/', num2str(split_ratio), '/');
 
        maxIter = 20;
        C_1 = 0.8;
        C_2 = 0.85;

        for k  = 1:num_sample
           %% ----- load the adj matrix and test data ----- 
           load(strcat(dataPath, 'sample_', num2str(k)));

           %% ----- initialization ------
           perSim  = cell(num_user, 1);
           userSim = zeros(num_user);

           %% ---- similarity propagation ----
           for u = 1 : num_user
               S = eye(num_item);
            
               W = full(trainAdj{u});
               W = W - diag(diag(W)) + eye(num_item);
 
               % normalize each column of the adj matrix W
               W = norm_by_col(W);

               for iter = 1 : maxIter
                   S = C_2 * W' * S * W;
                   S = S - diag(diag(S)) + eye(num_item);
                   S = S .* trainAdjZero{u} + full(trainAdjOne{u}) .* full(trainAdj{u});
               end
                
               perSim{u} = S;
           end
           
           %% compute the distance matrix dist
           dist = zeros(num_user);
           for u = 1 : (num_user - 1)
               for u_ = (u + 1) : num_user
                   froNorm = norm(perSim{u} - perSim{u_}, 'fro');
                   dist(u, u_) = froNorm;
                   dist(u_, u) = froNorm;
               end
           end
           
           for num_cluster = 1:num_user
               fprintf('number of cluster: %d', num_cluster)
               [inds, ~] = kmedioids(dist, num_cluster);
               
               % create a merged graph for each cluster
               adjMat  = cell(num_cluster, 1);
               
               for clusterId = 1:num_cluster
                   W = zeros(num_item);
                   members = find(inds == clusterId);
                   
                   for m = members
                       W = W + trainAdj{m};  
                   end
                   W = (W > 0);
                   adjMat{clusterId}  = (W > 0);
               end
               
               %% Step I: ----- initialization -----
                simCluster     = eye(num_cluster);
                perClusterSim  = cell(num_cluster, 1);
                for u = 1 : num_user
                   perClusterSim{u} =  eye(num_item);
                end 

               %% STEP II: ---- personalized simrank  ---
               for t =  1 : maxIter 
                   perClusterSim_t  = perClusterSim; 
                   simCluster_t     = simCluster;

                   % -- update  the personalized similarities --
                   for u = 1 : num_cluster
                       S = zeros(num_item);

                       for u_ = 1 : num_cluster
                           %W - adj matrix of u'
                           W = full(adjMat{u_}); 
                           W = W - diag(diag(W)) + eye(num_item);

                           % normalized column Adj matrix W
                           W = norm_by_col(W);

                           % update equation
                           S = S + C_1/num_cluster * simCluster_t(u, u_) * C_2 * W' * perClusterSim_t{u_} * W;
                       end

                       % -- diagonal elements are 1s --
                       S = S -  diag(diag(S))    + eye(num_item);
                       
                       % -- update the similarity matrix -
                       perClusterSim{u} = S;
                   end
                   
                   for u   = 1 : (num_cluster - 1)
                        for u_ = (u + 1) : num_cluster
                            froNorm = norm(perClusterSim{u} - perClusterSim{u_}, 'fro')/num_item;
                            simCluster(u, u_) = 1 - froNorm;
                            simCluster(u_, u) = 1 - froNorm;
                        end
                   end
               end
               
               for clusterId = 1:num_cluster
                   members = find(inds == clusterId);
                   for m = members
                       perSim{m} = perClusterSim{clusterId};  
                   end
               end
			
			%%-- Evaluation---
            recall = eval_recall(perSim, trainCluster, testCluster, num_user, num_item);
            pres   = eval_pres(perSim, trainCluster, testCluster, num_user, num_item);
            disp([recall, pres])
           end
        end
    end
end