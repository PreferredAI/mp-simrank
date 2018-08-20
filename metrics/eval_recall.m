function recall = eval_recall(per_sim, trainCluster, testCluster, num_user, num_item)
    recall = 0;
    for u = 1 : num_user
        s = per_sim{u}; 
        trCluster  = trainCluster{u};
        tCluster   = testCluster{u};
        
        u_recall = 0;
        count = 0;
        for gid = 1 : length(trCluster)
            trItems = trCluster{gid};
            tItems  = tCluster{gid};
            %disp(length(tItems))
            %disp(length(trItems))
            if (length(tItems) > 0 && (length(trItems) > 0)) %#ok<ISMT>
                count = count + 1;
                candidate = setdiff(1 : num_item, trItems);
                gsim    = s(trItems, candidate);
                gsim_score = sum(gsim);
                [~, topk]  = maxk(gsim_score, length(tItems));
                u_recall = u_recall + length(intersect(tItems, candidate(topk)))/length(tItems); 
            end
        end
        
        recall = recall + u_recall/count;
    end
    recall = recall/num_user;
end