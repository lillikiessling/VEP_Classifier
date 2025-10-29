function [ch3_noArti] = ArtifactRemovalFun(ch1, ch3, PulseDur,thresh)

    ch3 = ch3 - ch3(21);
    ch1 = ch1 - ch1(21);
%     if ch3(25) > 0
        flip = 1;
%     else
%         flip = -1
%     end
    ch1 = flip*ch1;
    %%try to remove device artifact from Ch3
    maxRange = 1:(21+PulseDur*2);
    [ch1_max,Ind] = max(abs(ch1(maxRange)));
    ch3_max = ch3(Ind);
    [~,ind3] = max(ch3(maxRange));
    if ind3-Ind >19
        ch3_noArti = ch3;
    else
        scale = (ch1_max-ch1(21))/(ch3_max-ch3(21));
        if thresh == 1
        ch3_noArti = ch3(21:21+PulseDur*2) - ch1(21:21+PulseDur*2)./scale;
        ch3_noArti = [ch3(1:20); ch3_noArti; ch3(22+PulseDur*2:end)];
        else           
        ch3_noArti = ch3 - ch1./scale;
    end

end