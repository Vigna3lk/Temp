clear; clc; close all;

%% 种子设置
rng('shuffle');
seed = rng;
disp(['实验种子：', num2str(seed.Seed)]);

% rng(1735744515);

%% 参数设置

% 雷达参数结构体
Radar_params = struct(...
    'R', 10, ...  % 雷达阵列圆周半径，m
    'RadarAmount', 4, ...  % 雷达阵列圆周上的阵元数（M）
    'c', 3e8, ...  % 光速
    'center', [0, 0, 0] ...
);

% 信源参数结构体
Src_params = struct(...
    'SrcAmount', 2, ...  % 信源数量
    'f_start', [1e3, 1e3], ...  % 信号的起始频率
    'f_end', [1e3, 1e3], ...    % 信号的终止频率
    'theta', [160, 60], ...  % 信源的俯仰角
    'phi', [50, 200], ...    % 信源的方向角
    'phase', [0, 0], ...     % 信源的初相位
    'SNR', 30, ...               % 信噪比
    'dt', 1e-5, ...              % 采样间隔
    'N', 2000 ...               % 拍数
);

t = 0:Src_params.dt:(Src_params.N-1)*Src_params.dt;  % 采样时间序列

%% 搜索范围
theta_st = 0;
theta_ed = 180;
theta_k = theta_st:theta_ed;
phi_st = 0;
phi_ed = 360;
phi_k = phi_st:phi_ed;

%% 1. 阵列信号生成

% 阵列位置向量:r_{m_i} = [R*cos(2*pi*(i-1)/M), R*sin(2*pi*(i-1)/M), 0]
theta = 2 * pi * (0:Radar_params.RadarAmount-1) / Radar_params.RadarAmount;
RadarPos = [Radar_params.R * cos(theta); Radar_params.R * sin(theta); zeros(1, Radar_params.RadarAmount)];

% 信源:[sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
h = [sind(Src_params.theta) .* cosd(Src_params.phi); ...
         sind(Src_params.theta) .* sind(Src_params.phi); ...
         cosd(Src_params.theta)];

% 阵元1对于其他阵元的延迟，缺少h的影响
% tau_ = RadarPos / Radar_params.c;
tau_ = (RadarPos(1, :) - RadarPos) / Radar_params.c;
tau = h' * tau_;

% 信源信号生成
src_signal = zeros(Src_params.SrcAmount, length(t));
for i = 1:Src_params.SrcAmount
    src_signal(i, :) = chirp(t, Src_params.f_start(i), t(end), Src_params.f_end(i), 'linear'); 
end

% 计算chirp信号的瞬时频率
f_inst = zeros(Src_params.SrcAmount, length(t));
for i = 1:Src_params.SrcAmount
    f_inst(i, :) = Src_params.f_start(i) + (Src_params.f_end(i) - Src_params.f_start(i)) * t / t(end);
end

% 延迟与信号叠加
signal = zeros(Radar_params.RadarAmount, length(t));
for i = 1:Src_params.SrcAmount
    for j = 1:Radar_params.RadarAmount
        % spatial_phase = exp(-1j * 2 * pi * 1e4 * tau(i, j));
        spatial_phase = exp(-1j * 2 * pi * f_inst(i, :) * tau(i, j));
        temporal_phase = src_signal(i, :);
        
        % 叠加信号
        signal(j, :) = signal(j, :) + spatial_phase .* temporal_phase;
    end
end

% 高斯白噪声
SNR_linear = 10^(Src_params.SNR / 10);  % 转换为线性
noise = (randn(size(signal)) + 1j * randn(size(signal))) / sqrt(2 * SNR_linear);  % 噪声
x = signal + noise;

%% 2. STFT
stft_winlen = 800;
window = hann(stft_winlen);
noverlap = 256;
fft_length = 1024;

X = zeros(Radar_params.RadarAmount, fft_length, floor((Src_params.N - noverlap) / (stft_winlen - noverlap)));
for k = 1:Radar_params.RadarAmount
    [X(k, :, :), f, t1] = stft(x(k, :), 1/Src_params.dt, 'Window', window, 'OverlapLength', noverlap, 'FFTLength', fft_length);
end


%% 3. 相关系数计算&筛选单源主导的时频区域
window = 4;
noverlap = 2;
G = floor((fft_length - noverlap) / (window - noverlap));  % 频域滑动窗口的数量

r = zeros(Radar_params.RadarAmount, Radar_params.RadarAmount, G);  % 相关系数
for i = 1:Radar_params.RadarAmount
    for j = i+1:Radar_params.RadarAmount
        for g = 1:G
            st = (g - 1) * (window - noverlap) + 1;
            ed = min(st + window - 1, fft_length);
            r(i, j, g) = abs(sum(sum(X(i, st:ed, :) .* conj(X(j, st:ed, :)))) / sqrt(sum(sum(abs(X(i, st:ed, :)).^2)) * sum(sum(abs(X(j, st:ed, :)).^2))));
        end
    end
end 
r = r + permute(r, [2, 1, 3]);  % 对称矩阵
for g = 1:G
    r(:, :, g) = eye(Radar_params.RadarAmount) + r(:, :, g);
end

mean_r = squeeze(mean(mean(r, 1), 2)); % 每个时频区域的平均相关系数
m = max(mean_r(:))  % 最大相关系数

% 筛选单源主导的时频区域
epsilon = 0.2;  % 相关系数阈值
single_source_regions = find(mean_r > 1 - epsilon);  % 单源主导的时频区域

%% 4. 搜索单源主导的时频区域, 估计DOA
doa = zeros(length(single_source_regions), 2);
Xi = zeros(theta_ed - theta_st + 1, phi_ed - phi_st + 1);
disp(['单源主导的时频区域数量：', num2str(length(single_source_regions))]);
for i = 1:length(single_source_regions)
    % fprintf(['\r', '正在搜索第', num2str(i), '个单源主导的时频区域...']);
    g = single_source_regions(i);
    st = (g - 1) * (window - noverlap) + 1;
    ed = min(st + window - 1, fft_length);

    g_f = f(st:ed);
    omega = 2 * pi * g_f;

    hat_X = X(:, st:ed, :);
    hat_X = reshape(hat_X, Radar_params.RadarAmount * length(g_f), []);
    xi = zeros(theta_ed - theta_st + 1, phi_ed - phi_st + 1);
    for th = theta_st:theta_ed
        for ph = phi_st:phi_ed
            h = [sind(th) .* cosd(ph); ...
                sind(th) .* sind(ph); ...
                cosd(th)];
            tau = h' * tau_;
            
            hat_A = compute_block_diagonal_matrix(omega, tau, Radar_params.RadarAmount);
            [Q, Lambda] = eig(hat_A);
            
            V_idx = find(int16(diff(diag(Lambda))) == 1);   % 由于lambda升序排列，V_idx实际上总为length(g_f) * (RadarAmount - 1)
            V = Q(:, 1:V_idx);
            xi(th - theta_st + 1, ph - phi_st + 1) = xi(th - theta_st + 1, ph - phi_st + 1) + 1 / norm(V' * hat_X, 'fro');
        end
    end
    fprintf('\n');
    disp(['第', num2str(i), '个单源主导的时频区域：', ' 相关系数：', num2str(mean_r(g)), '，范围：  ', num2str(st), ' - ', num2str(ed)]);

    [max_val, max_idx] = max(xi(:));
    [max_th, max_ph] = ind2sub(size(Xi), max_idx);
    doa(i, :) = [max_th + theta_st - 1, max_ph + phi_st - 1];

    disp(['最大值：', num2str(max_val), '在theta: ', num2str(doa(i, 1)), '°, phi: ', num2str(doa(i, 2)), '°']);
    % disp(['与真实值的误差：', num2str(norm(doa(i, :) - [Src_params.theta, Src_params.phi]))]);
    Xi = Xi + xi;
end

% doa估计结果
[max_val, max_idx] = max(Xi(:));
[max_th, max_ph] = ind2sub(size(Xi), max_idx);
t_doa = [max_th + theta_st - 1, max_ph + phi_st - 1];

% doa估计图
figure;
hold on;
scatter(doa(:, 1), doa(:, 2), '+', 'b');  % DOA 估计结果，蓝色 '+' 
scatter(Src_params.theta, Src_params.phi, 'o', 'r');  % 真实信源位置，红色 'o' 
scatter(t_doa(1), t_doa(2), 'x', 'g');  % 总体 DOA 估计结果，绿色 'x'
xlabel('theta (度)');
ylabel('phi (度)');
legend('DOA 估计', '真实信源位置', '总体 DOA 估计');
xlim([theta_st, theta_ed]);
ylim([phi_st, phi_ed]);
hold off;

function D = compute_block_diagonal_matrix(omega, tau, M)
    % 输入:
    % omega: 频率向量 [omega_1, omega_2, ..., omega_F]
    % tau: 时延矩阵 tau_{1,k}(theta_1), 大小为 1 x M
    % M: 向量 a 的长度
    %
    % 输出:
    % D: 分块对角矩阵

    F = length(omega); % 频率点的数量

    a = exp(1j * omega(:) * tau);

    D = zeros(M * F, M * F);
    for f = 1:F
        a_f = a(f, :).';                                % 提取当前频率的 a(omega_f, theta_1)
        D((f-1)*M+1:f*M, (f-1)*M+1:f*M) = (1/M) * (a_f * a_f' - eye(M)); % 计算并直接放入分块阵对应位置
    end
end