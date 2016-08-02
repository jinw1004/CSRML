% 速度已经提上来了，比预想的还要好，直接提了50多倍，在v4中，我们加入margin和regularization，当然首先要把input
% dim调大看看有没有性能上的提升
% v4里的参数是一组非常好的参数，如果后续没有什么提升的话可以直接使用这组参数来计算最终结果。
clear; clc;
demoDir = pwd;

addpath(genpath(fullfile(demoDir, '../')));
resDir = fullfile( demoDir, 'v6');
mkdir(resDir);
matDir = fullfile(demoDir, '..\mat\viper\');
partition = importdata(fullfile(matDir, 'partition_random.mat'));
% load feature
load('viper_lomo.mat'); % 34*1264
FeatG = descriptors(1:632, :)';
FeatP = descriptors(633:end, :)';
    
dim_input = 600;
dim_output = 300;

numClass = 632;
seed = 0;
rng(seed);

for trial = 1:10
%     p = randperm(numClass);
%     
%     trnSg = p(1:numClass/2);
%     trnSp = trnSg;
%     tstSg = p(numClass/2+1 : end);
%     tstSp = tstSg;
    trnSg = partition(trial).trnSg;
    trnSp = partition(trial).trnSp;
    tstSg = partition(trial).tstSg;
    tstSp = partition(trial).tstSp;

    FeatP_trn_ori = FeatP(:, trnSp);
    FeatG_trn_ori = FeatG(:, trnSg);
    FeatP_tst_ori = FeatP(:, tstSp);
    FeatG_tst_ori = FeatG(:, tstSg);
    FeatTrn_ori = [FeatP_trn_ori, FeatG_trn_ori];
    mu = mean(FeatTrn_ori, 2);
    % [W, ux] = pcaPmtk(FeatTrn_ori', dim_input); % W: ori-by-reduce
    options.ReducedDim = dim_input;
    [W, eigval] = myPCA(FeatTrn_ori', options);
%     clear feature FeatP FeatG FeatTrn;
    FeatP_trn = W'*(FeatP_trn_ori - repmat(mu, [1, size(FeatP_trn_ori, 2)]));
    FeatG_trn = W'*(FeatG_trn_ori - repmat(mu, [1, size(FeatG_trn_ori, 2)]));
    FeatP_tst = W'*(FeatP_tst_ori - repmat(mu, [1, size(FeatP_tst_ori, 2)]));
    FeatG_tst = W'*(FeatG_tst_ori - repmat(mu, [1, size(FeatG_tst_ori, 2)]));
    
    % exchange P and G
    tt = FeatP_trn; FeatP_trn = FeatG_trn; FeatG_trn = tt;
    tt = FeatP_tst; FeatP_tst = FeatG_tst; FeatG_tst = tt;
    
    % 把所有数据都放到Feat_trn里面去
    Feat_trn = zeros(size(FeatP_trn, 1), size(FeatP_trn, 2)+size(FeatG_trn, 2));
    Feat_trn(:, 1:2:end) = FeatP_trn;
    Feat_trn(:, 2:2:end) = FeatG_trn;
    % clear FeatP_trn FeatG_trn;

    L = eye(dim_output, dim_input);
    clear feature;


    % training
    info.train.loss = [];
    info.train.rank1 = []; % 这个以cmc rank-1作为error的指标么？
    info.val.loss = [];
    info.val.rank1 = [];

    % training options
    batchsize = 30;
    lr_g = [2e-4*ones(1, 300), 5e-5*ones(1, 50), 1e-5*ones(1, 50)];
    % lambda = 0.1;
    lambda = 0;
    beta = 6;
    tao = 1;
    m = 0.25;

%     L = gpuArray(L);
%     Feat_trn = gpuArray(Feat_trn);
    for epoch=1:300
        info.train.loss(end+1) = 0;
        info.train.rank1(end+1) = 0;
        info.val.loss(end+1) = 0;
        info.val.rank1(end+1) = 0;
    %     sum_grad = sum_grad_template; %相当于清零操作
        disp('====================================');

        id_trn = randperm(numel(trnSp));
        tic;
        a = 0;
        for t=1:batchsize:numel(trnSp)
            ksi = 0;
    %         sum_grad = sum_grad_template;

            batch = id_trn(t:min(t+batchsize-1, numel(trnSp)));
            idx_batch = [2*batch-1; 2*batch]; idx_batch = idx_batch(:)'; %[x;x*;]
            %获得输入图像（的特征）
            Xtrn_batch = Feat_trn(:, idx_batch);
            LXtrn_batch = L*Xtrn_batch;
            dist_batch = EuclidDist(LXtrn_batch', LXtrn_batch'); % gal-vs-prob
            dist_batch = dist_batch';

            % sample triplet
            id_batch = [1:numel(batch); 1:numel(batch)]; id_batch = id_batch(:)';
            tripletIndex = genTriplet(id_batch);


            % calculate in a batch mode
            Prb = tripletIndex(:,1);
            Pos = tripletIndex(:,2);
            Neg = tripletIndex(:,3);

            index_prb_pos = sub2ind(size(dist_batch), Prb, Pos);
            index_prb_neg = sub2ind(size(dist_batch), Prb, Neg);
            index_pos_neg = sub2ind(size(dist_batch), Pos, Neg);

            D_pos = dist_batch(index_prb_pos);
            D_neg = dist_batch(index_prb_neg);
            D_pos_neg = dist_batch(index_pos_neg);

            % calculate loss value
            Loss_pos = sum(1/beta * log(1+exp(beta*(m-tao+D_pos))), 1);
            Loss_neg = sum(1/beta * log(1+exp(beta*(m+tao-D_neg))), 1);
            Loss_pos_neg = sum(1/beta * log(1+exp(beta*(D_neg-D_pos_neg).^2)), 1);
            Loss = Loss_pos + Loss_neg + Loss_pos_neg;
            clear Loss_pos Loss_neg Loss_pos_neg;

            % calculate gradient
            OP1 = bsxfun(@times, 1./(1+exp(-beta*(m-tao+D_pos'))), LXtrn_batch(:,Prb) - LXtrn_batch(:,Pos));
            DZDL1 = OP1 * (Xtrn_batch(:,Prb) - Xtrn_batch(:,Pos))';
            clear OP1;

            OP2 = bsxfun(@times, -1./(1+exp(-beta*(m+tao-D_neg'))), LXtrn_batch(:,Prb) - LXtrn_batch(:,Neg));
            DZDL2 = OP2 * (Xtrn_batch(:,Prb) - Xtrn_batch(:,Neg))';
            clear OP2;

            coeff = 2*(D_pos_neg-D_neg)./(1+exp(-beta*(D_pos_neg-D_neg).^2));
            OP3 = bsxfun(@times, coeff', LXtrn_batch(:,Neg) - LXtrn_batch(:,Pos));
            OP4 = bsxfun(@times, coeff', LXtrn_batch(:,Neg) - LXtrn_batch(:,Prb));
            DZDL3 = OP3 * (Xtrn_batch(:,Neg) - Xtrn_batch(:,Pos))' - OP4 * (Xtrn_batch(:,Neg) - Xtrn_batch(:,Prb))';
            clear OP3 OP4;

            DZDL = DZDL1 + DZDL2 + DZDL3;
            clear DZDL1 DZDL2 DZDL3;

            fprintf('epoch %d, batch %d, loss %.4f\n', epoch, floor((t+batchsize)/batchsize), Loss/size(tripletIndex, 1));
            % update parameters
            lr = lr_g(epoch);
            L = L - lr*DZDL - lr*lambda*L;
            a = a + Loss;
        end
%         L = gather(L);
%         a = gather(a);
        epoch_time = toc;
        fprintf('epoch time %.2f\n', epoch_time);
        info.train.loss(end) = a;   

        % performance on training set
        dist_trn = EuclidDist(FeatG_trn'*L', FeatP_trn'*L');
        dist_trn = dist_trn';
        cmc_trn = evaluate_cmc(-dist_trn)/316;
        fprintf('*****VIPeR: epoch %d, training set cmc *****\n', epoch);
        fprintf('rank1\t\trank5\t\trank10\t\trank15\t\trank20 : \n %2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%% \n', ...
            100*cmc_trn(1), 100*cmc_trn(5), 100*cmc_trn(10), 100*cmc_trn(15), 100*cmc_trn(20));
        info.train.rank1(end) = cmc_trn(1);

        % performance on test set
        dist_tst = EuclidDist(FeatG_tst'*L', FeatP_tst'*L');
        dist_tst = dist_tst';
        cmc_tst = evaluate_cmc(-dist_tst)/316;
        fprintf('*****VIPeR: epoch %d, testing set cmc *****\n', epoch);
        fprintf('rank1\t\trank5\t\trank10\t\trank15\t\trank20 : \n %2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%% \n', ...
            100*cmc_tst(1), 100*cmc_tst(5), 100*cmc_tst(10), 100*cmc_tst(15), 100*cmc_tst(20));
        info.val.rank1(end) = cmc_tst(1);

        % plot the result
        hh=figure(1);clf;
        subplot(1,2,1);
        semilogy(1:epoch, info.train.loss, 'k-'); hold on;
        xlabel('epoch'); ylabel('loss'); h = legend('train'); grid on;
        set(h, 'color', 'none');
        title('total loss of train');

        subplot(1,2,2);
        plot(1:epoch, info.train.rank1, 'k-'); hold on;
        plot(1:epoch, info.val.rank1, 'g-');ylim([0 1]);
        xlabel('epoch'); ylabel('cmc rank1'); h = legend('train', 'val'); grid on;
        set(h, 'color', 'none');
        title('error');
        drawnow;

        if mod(epoch, 10) == 0
            savefig(hh, fullfile(resDir, ['viper_' num2str(trial) '.fig']));
            save(fullfile(resDir, ['viper_' num2str(epoch) '_' num2str(trial) '.mat']), 'cmc_tst');
        end
    %     pause(0.5);
    end
end