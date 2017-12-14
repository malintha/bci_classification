%Mu = 3-7, lowerFr=6Hz upperFr=14Hz,  Mu+Beta = 3-15,lowerFr=6Hz
%upperFr=30Hz ,ony set numbers which are devisible by 2
function[trial_mat] = dostft(x, window_size, overlap,windowName, lowerFr, upperFr)
[~, channels, trials] = size(x);
trial_mat = [];
for t=1:trials
    channel_vec = [];
    for c=1:channels
        channel = x(:,c,t);
        s = spectrogram(channel,getWindow(windowName,window_size),overlap);
        if rem(lowerFr,2)~=0
            lowerFr=lowerFr+1;
        end
        if rem(upperFr,2)~=0
            upperFr=upperFr+1;
        end
        sFiltered=s(lowerFr/2:upperFr/2,:);
        [m,n] = size(sFiltered);
%         v = reshape(sFiltered,[m*n 1]);
        channel_vec = [channel_vec; sFiltered];
    end
    trial_mat = [trial_mat channel_vec];
end

    %This function gets different windows:
    function window = getWindow(windowName,window_size)
       if strcmp(windowName,'hann')
           window=hann(window_size);
       elseif strcmp(windowName,'blackman')
           window=blackman(window_size);   
       elseif strcmp(windowName,'hamming')
           window=hamming(window_size);   
       elseif strcmp(windowName,'gausswin')
           window=gausswin(window_size);   
       else
           window=null;
       end         
    end

end