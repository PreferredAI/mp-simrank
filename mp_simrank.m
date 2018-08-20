addpath(genpath(['MinMaxSelection']));
addpath(genpath(['utils']));
addpath(genpath(['metrics']));

%% create neighbor (label: 1) and non-neighbor (label: -1)graphs from pairs of objects
dataset = 'pcc';
split_ratios = [0.7];
num_sample   = 10;

%for dataset = data_names
    disp(strcat('--- dataset = ', dataset , '-----'))
    for split_ratio = split_ratios
        disp(strcat('--- split-ratio = ', num2str(split_ratio) , '-----'))
		%path_to_your_data
        dataPath = strcat('data/', dataset, '/', num2str(split_ratio), '/');
        
        maxIter = 30; 
        C_1 = 1;
        C_2 = 0.85;
        disp('persim: 1 - fro_norm')
        for k  = 1 : num_sample
           %% ----- load the adj matrix and test data ----- 
           load(strcat(dataPath, 'sample_', num2str(k)));

           %% Step I: ----- initialization -----
            simUser = eye(num_user);
            perSim  = cell(num_user, 1);
            for u = 1 : num_user
               perSim{u} =  eye(num_item);
            end 

           %% STEP II: ---- personalized simrank  ---
           for t =  1 : maxIter 
               
               perSim_t  = perSim; 
               simUser_t = simUser;
               
               % -- update  the personalized similarities --
               for u = 1 : num_user
                   S = zeros(num_item);

                   for u_ = 1 : num_user
                       %W - adj matrix of u'
                       W = full(trainAdj{u_}); 
                       W = W - diag(diag(W)) + eye(num_item);

                       % normalized column Adj matrix W
                       W = norm_by_col(W);

                       % update equation
                       S = S + C_1/num_user * simUser_t(u, u_) * C_2 * W' * perSim_t{u_} * W;
                   end

                   % -- diagonal elements are 1s --
                   S = S -  diag(diag(S))    + eye(num_item);
                   S = S .* trainAdjZero{u} + full(trainAdjOne{u}) .* full(trainAdj{u});
                   
                   % -- update the similarity matrix -
                   perSim{u} = S;
               end
               %% --- convergence analysis
               D = 0;
               for u = 1:num_user
                   D = D + norm(perSim{u} - perSim_t{u}, 'fro')/(num_item * num_user);
               end
               disp(D)
               
               %% --- update the user-user similarity matrix ---
               for u = 1 : (num_user - 1)
                   for u_ = (u + 1) : num_user
                       froNorm = norm(perSim{u} - perSim{u_}, 'fro')/num_item;
                        
                       simUser(u, u_) = 1 - froNorm;
                       simUser(u_, u) = 1 - froNorm;
                   end
               end
           end
           

           %% --- Evaluation ---  
           recall = eval_recall(perSim, trainCluster, testCluster, num_user, num_item);
           pres   = eval_pres(perSim, trainCluster, testCluster, num_user, num_item);
           disp([recall, pres])
           
        end
    end
%end
