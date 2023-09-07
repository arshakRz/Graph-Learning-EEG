clc; close all; clear all; 
load('Ex3.mat');
load('AllElectrodes.mat');
name = ["AFz", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO3", "PO4", "O1", "O2"];
all_labels = cell(64, 1);
for i = 1:64
    all_labels(i) = cellstr(AllElectrodes(i).labels);
end
coords = zeros(30, 2);
for i = 1:length(name)
    ind = find(all_labels==name(i));
    coords(i, :) = [AllElectrodes(ind).X AllElectrodes(ind).Y];
end

dist = zeros(30, 30);
for i = 1:length(name)
    for j = 1:length(name)
    ind1 = find(all_labels==name(i));
    ind2 = find(all_labels==name(j));
    vec1 = [AllElectrodes(ind1).X AllElectrodes(ind1).Y AllElectrodes(ind1).Z];
    vec2 = [AllElectrodes(ind2).X AllElectrodes(ind2).Y AllElectrodes(ind2).Z];
    dist(i, j) = sqrt(sum((vec1-vec2).^2));
    if i == j 
        dist(i, j) = 1;
    end
    end
end
dist = 1./dist;
dist = dist - diag(diag(dist));
dist = dist/max(dist,[],'all');
dist(dist>0.25) = 0;
load("embeddings.mat")
%%
clc; 
n_channels = size(TrainData,1);
%n_points = size(TrainData,2);
n_trials = size(TrainData,3);
Fs = 256;
t = 1/Fs:1/Fs:1;
%% plotting one VG
ds = round(linspace(1, size(TrainLabel, 2), 20));
n = 1;     
trial = TrainData(:, :, n);
i = 1;
time_series = trial(i, :);
VG = fast_NVG(time_series(ds), t(ds), 'w', 0);
figure
stem(t(ds), time_series(ds));
hold on
for q = 1:size(VG, 1)
    for p = 1:size(VG, 2)
        if ~VG(q, p)==0
            a = VG(q, p);
            plot([t(ds(q)) t(ds(p))], [time_series(ds(q)) time_series(ds(p))])
        end
    end
end
xlabel("t(s)", "Interpreter","latex")
title("The Visibility Graph On A Time Series", 'Interpreter','latex')
figure
stem(t(ds), time_series(ds));
xlabel("t(s)", "Interpreter","latex")
title("A Time Series", 'Interpreter','latex')
%%
feat = zeros(n_trials, n_channels);
for n = 1:n_trials
    trial = TrainData(:, :, n);
    for i = 1:n_channels
        time_series = trial(i, :);
        VG = fast_NVG(time_series, t, 'w', 0);
        G = gsp_graph(VG);
        G = gsp_create_laplacian(G);
        G = gsp_compute_fourier_basis(G);
        X_GDFT = G.U'*time_series';
        feat(n, i) = sum(abs(X_GDFT).^2)/length(X_GDFT);
    end
    n
end
%% 
clc; close all;
feat_norm = mapstd(feat', 0, 1);
feat_norm = feat_norm';
%%
idx = fscmrmr(feat_norm,TrainLabel);
feat_mrmr_norm = feat_norm(:,idx(1));
%%
C = zeros([3 length(TrainLabel)]);
C(3,TrainLabel==1) = 1;
C(1,TrainLabel==0) = 1;
figure 
scatter(feat_mrmr_norm(:,1),2,40,C');
grid minor
xlabel("$x_1$",'Interpreter','latex')
%%
X = feat_mrmr_norm;
Y = TrainLabel;
cvp = cvpartition(Y, 'KFold', 5);
mdl = fitcsvm(X, Y, "KernelFunction", "rbf", 'CVPartition',cvp);
prediction = kfoldPredict(mdl);
figure
plotconfusion(Y, prediction')

%%
clc; close all;
n_channels = size(TrainData,1);
feat = zeros(n_trials,n_channels+n_channels*(n_channels-1)/2);
n_trials = size(TrainData,3);
for n = 1:n_trials
    trial = TrainData(:, :, n);
    cor_data = corrcoef(trial');
    m = 1;
    for j = 1:30
        for k = j+1:30
            up(m) = cor_data(k,j);
            m = m+1;
        end
    end
    feat(n, 1:n_channels) = eig(cor_data);
    feat(n, (n_channels+1):end) = up;
end
imagesc(cor_data)
set(gca, 'XTick', [], 'YTick', []);
%%
clc;
n_channels = size(TrainData,1);
n_points = size(TrainData,2);
n_trials = size(TrainData,3);
%%
% Calculating the AR coeffs
n_ar_coeff = 21;
data_ar_coeff = zeros([n_channels, n_ar_coeff, n_trials]);
for n = 1:n_trials
    trial = TrainData(:, :, n);
    for i = 1:n_channels
        time_series = trial(i, :);
        ar_time_series = ar(time_series,n_ar_coeff);
        data_ar_coeff(i, :, n) = ar_time_series.A(2:end);
    end
    n
end
%%
% Calculating the ARIMA coeffs
n_arima_coeff = 5;
Mdl = arima(3,1,2);
data_arima_coeff = zeros([n_channels, n_arima_coeff, n_trials]);
for n = 1:n_trials
    trial = TrainData(:, :, n);
    for i = 1:n_channels
        time_series = trial(i, :);
        arima_time_series = estimate(Mdl,time_series');
        data_arima_coeff(i, :, n) = cell2mat([arima_time_series.AR arima_time_series.MA]);
    end
end
%%
clc; close all;
feat = zeros(n_trials,n_channels+n_channels*(n_channels-1)/2);
Gs = [];
Ws = zeros(165, 30, 30);
for n = 1:n_trials
    %trial = TrainData(:, :, n);
    trial_ar_coeff = data_ar_coeff(:, :, n);
    %trial_arima_coeff = data_arima_coeff(:, :, n);
    %z = gsp_distanz(trial_arima_coeff').^2;
    z = gsp_distanz(trial_ar_coeff').^2;
    %z = gsp_distanz(trial').^2;
    theta = gsp_compute_graph_learning_theta(z, 10);
    W = gsp_learn_graph_log_degrees(z * theta, 1, 1);
    W = W/max(W,[],'all');
    W(W<0.1) = 0;
    Ws(n, :, :) = W;
    G = gsp_graph(W, [coords(:, 2), coords(:, 1)]);
    G = gsp_create_laplacian(G);
    G = gsp_compute_fourier_basis(G);
    m = 1;
    for j = 1:30
        for k = j+1:30
            up(m) = G.W(k,j);
            m = m+1;
        end
    end
    %feat(n, :) = vecnorm(diff(G.U,1,1), 2, 1);
    feat(n, 1:n_channels) = G.e;
    feat(n, (n_channels+1):end) = up;
    Gs = [Gs G];
end

W1 = zeros([30 30]);
W2 = zeros([30 30]);

for i = 1:length(Gs)
    G = Gs(i);
    if TrainLabel(i) == 0
        W1 = W1 + G.W;
    else
        W2 = W2 + G.W;
    end   
end


W1 = W1/sum(TrainLabel==0);
W2 = W2/sum(TrainLabel==1);

figure
plotGraph(W1, coords, 1.5)
set(gca, 'XTick', [], 'YTick', []);
title("Average Sparsed Graph For Class One", "Interpreter", "latex")
figure
plotGraph(W2, coords, 1.5)
set(gca, 'XTick', [], 'YTick', []);
title("Average Sparsed Graph For Class Two", "Interpreter", "latex")
%%
feat_norm = mapstd(feat', 0, 1);
feat_norm = feat_norm';
feat = feat_norm;
%%
idx = fscmrmr(feat,TrainLabel);
feat_mrmr = feat(:,idx(1:10));
%%
figure 
C = zeros([3 length(TrainLabel)]);
C(3,TrainLabel==1) = 1;
C(1,TrainLabel==0) = 1;
scatter3(feat_mrmr(:,1),feat_mrmr(:,2),feat_mrmr(:,3),40,C',"filled");
xlabel("$x_1$",'Interpreter','latex')
ylabel("$x_2$",'Interpreter','latex')
zlabel("$x_3$",'Interpreter','latex')

%%
X = feat_mrmr;
Y = TrainLabel;
cvp = cvpartition(Y, 'KFold', 5);
mdl = fitcensemble(X, Y, 'Method', 'AdaBoostM1', 'CVPartition', cvp);
prediction = kfoldPredict(mdl);
figure
plotconfusion(Y, prediction',"5 Fold Crossvalidation")



