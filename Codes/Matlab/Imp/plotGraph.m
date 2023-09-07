function [] = plotGraph(W, coords, alpha)
    scatter(coords(:,2),coords(:,1),40,"black",'filled');
    xlim([-100 100])   
    ylim([-100 100])
    hold on
    n = length(coords);
    W(W<mean(mean(W))*alpha) = 0;
    for i = 1:n
        for j = (i+1):n
            if (W(i, j)~=0)
                plot([coords(i, 2), coords(j, 2)], [coords(i, 1), coords(j, 1)],"color",'black',"LineWidth",2*W(i,j)/max(max(W)))
            end
        end
    end
end