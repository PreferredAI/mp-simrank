addpath(genpath(['MinMaxSelection']));
addpath(genpath(['utils']));
addpath(genpath(['metrics']));


%% create neighbor (label: 1) and non-neighbor (label: -1)graphs from pairs of objects
dataset = 'pcc';
split_ratios = [0.7];
num_sample   = 10;

disp('--- pipelined_simrank ---')
%for dataset = data_names
    disp(strcat('--- dataset = ', dataset , '-----'))
    for split_ratio = split_ratios
        disp(strcat('--- split-ratio = ', num2str(split_ratio) , '-----'))
        dataPath = strcat('data/', dataset, '/', num2str(split_ratio), '/');
        
        maxIter = 30;
        C   = 1;
        C_2 = 0.85;

        for k  = 1:num_sample
           %% ----- load the adj matrix and test data ----- 
           load(strcat(dataPath, 'sample_', num2str(k)));
           num_pair = size(user_pairObj, 2);

           %% ----- initialization ------
           perSim  = cell(num_user, 1);
           for u = 1 : num_user
             perSim{u} =  eye(num_item);
           end 

           simUser = eye(num_user);
           simPair = eye(num_pair);

           %% --- user similarity with bipartie simrank ---
           for iter = 1 : maxIter
               simUser_t = simUser;
               simPair_t = simPair;

               W = full(user_pairObj);
               cnorm_W = norm_by_col(W);
               rnorm_W = norm_by_col(W');

               simUser  = C * rnorm_W' * simPair_t * rnorm_W;
               simPair  = C * cnorm_W' * simUser_t * cnorm_W; 
               
               simUser = simUser - diag(diag(simUser)) + eye(num_user);
               simPair = simPair - diag(diag(simPair)) + eye(num_pair);
           end
           simUser = simUser - diag(diag(simUser)) + eye(num_user);
           
           %% --- similarity propagation ----
           for iter = 1 : maxIter
               perSim_t = perSim;
               for u = 1 : num_user
                   S = zeros(num_item);

                   for u_ = 1 : num_user
                       % W - adj matrix of u'
                       W_2 = full(trainAdj{u_}); 
                       W_2 = W_2 - diag(diag(W_2)) + eye(num_item);
                       W_2 = norm_by_col(W_2);

                       % update equation
                       S   = S + 1/num_user * simUser(u, u_) * C_2 * W_2' * perSim_t{u_} * W_2;
                   end

                   % -- diagonal elements are 1s --
                   S = S - diag(diag(S)) + eye(num_item);
                   S = S .* trainAdjZero{u} + full(trainAdjOne{u}) .* full(trainAdj{u});
                   
                   % -- update the similarity matrix -
                   perSim{u} = S;
               end
               
               D = 0;
               for u = 1:num_user
                   D = D + norm(perSim{u} - perSim_t{u}, 'fro')/(num_item * num_user);
               end
               disp(D)
           end
            
           
           %% ---- evaluation ----
           recall = eval_recall(perSim, trainCluster, testCluster, num_user, num_item);
           pres   = eval_pres(perSim, trainCluster, testCluster, num_user, num_item);
           disp([recall, pres]);
        end
    end
%end