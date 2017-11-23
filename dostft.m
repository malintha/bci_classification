function[trial_mat] = dostft(x, window_size, overlap)
[~, channels, trials] = size(x);
trial_mat = [];
for t=1:trials
    channel_vec = [];
    for c=1:channels
        channel = x(:,c,t);s
        s = spectrogram(channel,hann(window_size),overlap);
        [m,n] = size(s);
        v = reshape(s,[m*n 1]);
        channel_vec = [channel_vec; v];
    end
    trial_mat = [trial_mat channel_vec];
end
end