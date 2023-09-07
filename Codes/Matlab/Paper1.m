%% Signal Generation 
clc; close all; clear all; 
%% Coupled Broad Band Signals
c = 0.9;
n1= randn(1,1000);
n2= randn(1,1000);
n3= randn(1,1000);
x1 = (1-c) * n1 + c * n3;
x2 = (1-c) * n2 + c * n3;
%% Coupled Narrow Band Signals:
fs = 1000;
t = 1/fs:1/fs:1;
df = (fs)/(length(t)-1);
f = -fs/2:df:fs/2;
f0 = 50;
n1= randn(1,length(t));
n2= randn(1,length(t));
n3= randn(1,length(t));
n4= randn(1,length(t));
nf1 = lowpass(n1,10,fs);
nf2 = lowpass(n2,10,fs);
nf3 = lowpass(n3,10,fs);
nf4 = lowpass(n4,10,fs);
A1 = sqrt(nf1.^2+nf2.^2);
A2 = sqrt(nf3.^2+nf4.^2);
phi1 = atan(nf2./nf1);
phi2 = atan(nf4./nf3);
%% PR:
c = 0.9;
x1 = A1 .* cos(2*pi*f0*t+phi1);
x2 = A2 .* cos(2*pi*f0*t+ c * phi1 + (1-c) * phi2);
%% AR:
c = 0.9;
x1 = A1 .* cos(2*pi*f0*t+phi1);
x2 = (c * A1 + (1-c) * A2) .* cos(2*pi*f0*t+phi2);

%% Regression Methods
clc; close all; 

% R^2
R = corrcoef(x1,x2);

% CF
CF = mscohere(x1,x2,[],[],f,fs);
plot(CF)

% h^2 
%% Phase Synchronization Methods
clc; close all; 

% Using Hilbert Transform
x1h = hilbert(x1);
x2h = hilbert(x2);
z1 = x1+ 1i*x1h;
z2 = x2+ 1i*x2h;
phi1 = angle(z1);
phi2 = angle(z2);
phi = mod(phi1-phi2, 2*pi);

% Shannon Entropy
edges = 0:0.32:6.4;
p = histcounts(phi, edges)/1000;
rho = 1 + 1/log(length(p)) * sum(p(p~=0).*log(p(p~=0)));

% Mean Phase Coherence
R = abs(1/length(phi) * sum(exp(1i*phi)));
%%