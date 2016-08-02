clear CMC;
resDir = fullfile(pwd, 'v6');
for i=1:10
    name = ['viper_300_' num2str(i) '.mat'];
    load(fullfile(resDir, name));
    CMC(i,:) = cmc_tst;
end
cmc2 = mean(CMC, 1);
fprintf('*****VIPeR average cmc *****\n');
fprintf('rank1\t\trank5\t\trank10\t\trank15\t\trank20 : \n %2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%% \n', ...
    100*cmc2(1), 100*cmc2(5), 100*cmc2(10), 100*cmc2(15), 100*cmc2(20));

% plot the results
figure(1);
%plot(1:30, cmc(1:30), 'k', 'LineWidth', 2); hold on;
plot(1:30, cmc2(1:30), 'g', 'LineWidth', 2); hold on;
xlim([1 30]); ylim([0.3 1]); grid on;