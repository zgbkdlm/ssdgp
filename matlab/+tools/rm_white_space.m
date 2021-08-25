function rm_white_space()
% Remove white space of Matlab figure
% Source: https://gist.github.com/zgbkdlm/f7663c1dfaea0716ddaaf2b89dff7bd7
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

% However, the above cropping might very possibly cause a warning error when saving the figure,
% and there will be a small part of your figure being over-cropped. This can be simply
% solved by runing this

set(gcf, 'PaperUnits', 'normalized')
set(gcf, 'PaperPosition', [0 0 1 1])

end

