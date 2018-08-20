function pres = eval_pres(perSim, trainCluster, testCluster, num_user, num_item)
    pres = 0; 
    nz_num_user = num_user;
    for u = 1 : num_user
        s = perSim{u}; 
        trCluster  = trainCluster{u};
        tCluster   = testCluster{u};
        
        u_pres = 0;
        count = 0;
        for gid = 1 : length(trCluster)
            trItems = trCluster{gid};
            tItems  = tCluster{gid};
            ntItems = length(tItems);
            if (length(tItems) > 0 && (length(trItems) > 0)) %#ok<ISMT>
                count = count + 1;
                candidate  = setdiff(1 : num_item, trItems);
                gsim       = s(trItems, candidate);
                gsim_score = sum(gsim);
                [~, topk]  = maxk(gsim_score, length(candidate));
                rankedCandidate = candidate(topk);
                position   = find(ismember(rankedCandidate, tItems));
                u_pres = u_pres + 1 - (sum(position)/ntItems - (ntItems + 1)/2)/(num_item - length(trItems));
            end
        end
        
        if count ~= 0
            pres = pres + u_pres/count;   
        else
            nz_num_user = nz_num_user - 1;
        end
    end
    pres = pres/nz_num_user;
end