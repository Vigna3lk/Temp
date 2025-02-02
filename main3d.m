clear; clc; close all;

%% ��������
rng('shuffle');
seed = rng;
disp(['ʵ�����ӣ�', num2str(seed.Seed)]);

% rng(1735744515);

%% ��������

% �״�����ṹ��
Radar_params = struct(...
    'R', 10, ...  % �״�����Բ�ܰ뾶��m
    'RadarAmount', 4, ...  % �״�����Բ���ϵ���Ԫ����M��
    'c', 3e8, ...  % ����
    'center', [0, 0, 0] ...
);

% ��Դ�����ṹ��
Src_params = struct(...
    'SrcAmount', 2, ...  % ��Դ����
    'f_start', [1e3, 1e3], ...  % �źŵ���ʼƵ��
    'f_end', [1e3, 1e3], ...    % �źŵ���ֹƵ��
    'theta', [160, 60], ...  % ��Դ�ĸ�����
    'phi', [50, 200], ...    % ��Դ�ķ����
    'phase', [0, 0], ...     % ��Դ�ĳ���λ
    'SNR', 30, ...               % �����
    'dt', 1e-5, ...              % �������
    'N', 2000 ...               % ����
);

t = 0:Src_params.dt:(Src_params.N-1)*Src_params.dt;  % ����ʱ������

%% ������Χ
theta_st = 0;
theta_ed = 180;
theta_k = theta_st:theta_ed;
phi_st = 0;
phi_ed = 360;
phi_k = phi_st:phi_ed;

%% 1. �����ź�����

% ����λ������:r_{m_i} = [R*cos(2*pi*(i-1)/M), R*sin(2*pi*(i-1)/M), 0]
theta = 2 * pi * (0:Radar_params.RadarAmount-1) / Radar_params.RadarAmount;
RadarPos = [Radar_params.R * cos(theta); Radar_params.R * sin(theta); zeros(1, Radar_params.RadarAmount)];

% ��Դ:[sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
h = [sind(Src_params.theta) .* cosd(Src_params.phi); ...
         sind(Src_params.theta) .* sind(Src_params.phi); ...
         cosd(Src_params.theta)];

% ��Ԫ1����������Ԫ���ӳ٣�ȱ��h��Ӱ��
% tau_ = RadarPos / Radar_params.c;
tau_ = (RadarPos(1, :) - RadarPos) / Radar_params.c;
tau = h' * tau_;

% ��Դ�ź�����
src_signal = zeros(Src_params.SrcAmount, length(t));
for i = 1:Src_params.SrcAmount
    src_signal(i, :) = chirp(t, Src_params.f_start(i), t(end), Src_params.f_end(i), 'linear'); 
end

% ����chirp�źŵ�˲ʱƵ��
f_inst = zeros(Src_params.SrcAmount, length(t));
for i = 1:Src_params.SrcAmount
    f_inst(i, :) = Src_params.f_start(i) + (Src_params.f_end(i) - Src_params.f_start(i)) * t / t(end);
end

% �ӳ����źŵ���
signal = zeros(Radar_params.RadarAmount, length(t));
for i = 1:Src_params.SrcAmount
    for j = 1:Radar_params.RadarAmount
        % spatial_phase = exp(-1j * 2 * pi * 1e4 * tau(i, j));
        spatial_phase = exp(-1j * 2 * pi * f_inst(i, :) * tau(i, j));
        temporal_phase = src_signal(i, :);
        
        % �����ź�
        signal(j, :) = signal(j, :) + spatial_phase .* temporal_phase;
    end
end

% ��˹������
SNR_linear = 10^(Src_params.SNR / 10);  % ת��Ϊ����
noise = (randn(size(signal)) + 1j * randn(size(signal))) / sqrt(2 * SNR_linear);  % ����
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


%% 3. ���ϵ������&ɸѡ��Դ������ʱƵ����
window = 4;
noverlap = 2;
G = floor((fft_length - noverlap) / (window - noverlap));  % Ƶ�򻬶����ڵ�����

r = zeros(Radar_params.RadarAmount, Radar_params.RadarAmount, G);  % ���ϵ��
for i = 1:Radar_params.RadarAmount
    for j = i+1:Radar_params.RadarAmount
        for g = 1:G
            st = (g - 1) * (window - noverlap) + 1;
            ed = min(st + window - 1, fft_length);
            r(i, j, g) = abs(sum(sum(X(i, st:ed, :) .* conj(X(j, st:ed, :)))) / sqrt(sum(sum(abs(X(i, st:ed, :)).^2)) * sum(sum(abs(X(j, st:ed, :)).^2))));
        end
    end
end 
r = r + permute(r, [2, 1, 3]);  % �Գƾ���
for g = 1:G
    r(:, :, g) = eye(Radar_params.RadarAmount) + r(:, :, g);
end

mean_r = squeeze(mean(mean(r, 1), 2)); % ÿ��ʱƵ�����ƽ�����ϵ��
m = max(mean_r(:))  % ������ϵ��

% ɸѡ��Դ������ʱƵ����
epsilon = 0.2;  % ���ϵ����ֵ
single_source_regions = find(mean_r > 1 - epsilon);  % ��Դ������ʱƵ����

%% 4. ������Դ������ʱƵ����, ����DOA
doa = zeros(length(single_source_regions), 2);
Xi = zeros(theta_ed - theta_st + 1, phi_ed - phi_st + 1);
disp(['��Դ������ʱƵ����������', num2str(length(single_source_regions))]);
for i = 1:length(single_source_regions)
    % fprintf(['\r', '����������', num2str(i), '����Դ������ʱƵ����...']);
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
            
            V_idx = find(int16(diff(diag(Lambda))) == 1);   % ����lambda�������У�V_idxʵ������Ϊlength(g_f) * (RadarAmount - 1)
            V = Q(:, 1:V_idx);
            xi(th - theta_st + 1, ph - phi_st + 1) = xi(th - theta_st + 1, ph - phi_st + 1) + 1 / norm(V' * hat_X, 'fro');
        end
    end
    fprintf('\n');
    disp(['��', num2str(i), '����Դ������ʱƵ����', ' ���ϵ����', num2str(mean_r(g)), '����Χ��  ', num2str(st), ' - ', num2str(ed)]);

    [max_val, max_idx] = max(xi(:));
    [max_th, max_ph] = ind2sub(size(Xi), max_idx);
    doa(i, :) = [max_th + theta_st - 1, max_ph + phi_st - 1];

    disp(['���ֵ��', num2str(max_val), '��theta: ', num2str(doa(i, 1)), '��, phi: ', num2str(doa(i, 2)), '��']);
    % disp(['����ʵֵ����', num2str(norm(doa(i, :) - [Src_params.theta, Src_params.phi]))]);
    Xi = Xi + xi;
end

% doa���ƽ��
[max_val, max_idx] = max(Xi(:));
[max_th, max_ph] = ind2sub(size(Xi), max_idx);
t_doa = [max_th + theta_st - 1, max_ph + phi_st - 1];

% doa����ͼ
figure;
hold on;
scatter(doa(:, 1), doa(:, 2), '+', 'b');  % DOA ���ƽ������ɫ '+' 
scatter(Src_params.theta, Src_params.phi, 'o', 'r');  % ��ʵ��Դλ�ã���ɫ 'o' 
scatter(t_doa(1), t_doa(2), 'x', 'g');  % ���� DOA ���ƽ������ɫ 'x'
xlabel('theta (��)');
ylabel('phi (��)');
legend('DOA ����', '��ʵ��Դλ��', '���� DOA ����');
xlim([theta_st, theta_ed]);
ylim([phi_st, phi_ed]);
hold off;

function D = compute_block_diagonal_matrix(omega, tau, M)
    % ����:
    % omega: Ƶ������ [omega_1, omega_2, ..., omega_F]
    % tau: ʱ�Ӿ��� tau_{1,k}(theta_1), ��СΪ 1 x M
    % M: ���� a �ĳ���
    %
    % ���:
    % D: �ֿ�ԽǾ���

    F = length(omega); % Ƶ�ʵ������

    a = exp(1j * omega(:) * tau);

    D = zeros(M * F, M * F);
    for f = 1:F
        a_f = a(f, :).';                                % ��ȡ��ǰƵ�ʵ� a(omega_f, theta_1)
        D((f-1)*M+1:f*M, (f-1)*M+1:f*M) = (1/M) * (a_f * a_f' - eye(M)); % ���㲢ֱ�ӷ���ֿ����Ӧλ��
    end
end