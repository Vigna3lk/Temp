clear; clc; close all;

% rng('shuffle');
% seed = rng;
% disp(['ʵ�����ӣ�', num2str(seed.Seed)]);

rng(1735744515);

% �����ṹ��
Radar_params = struct(...
    'R', 100, ...  % �״�����Բ�ܰ뾶��m
    'RadarAmount', 4, ...  % �״�����Բ���ϵ���Ԫ����M��
    'c', 3e8, ...  % ����
    'center', [0, 0] ...  % 2D���ĵ�
);

Src_params = struct(...
    'SrcAmount', 2, ...  % ��Դ����
    'f_start', [1e3, 1e4], ...  % �źŵ���ʼƵ��
    'f_end', [1e3, 1e4], ...    % �źŵ���ֹƵ��
    'phi', [10, 50], ...    % ��Դ�ķ����(�ڶ�άƽ����)
    'phase', [0, 0], ...     % ��Դ�ĳ���λ
    'SNR', 30, ...           % �����
    'dt', 1e-5, ...          % �������
    'N', 2000 ...           % ����
);

t = 0:Src_params.dt:(Src_params.N-1)*Src_params.dt;  % ����ʱ������
t_matrix = repmat(t, Radar_params.RadarAmount, 1);

phi_st = 0;
phi_ed = 360;
phi_k = phi_st:phi_ed;

% ����λ������(��ά): r_{m_i} = [R*cos(2*pi*(i-1)/M), R*sin(2*pi*(i-1)/M)]
theta = 2 * pi * (0:Radar_params.RadarAmount-1) / Radar_params.RadarAmount;
RadarPos = [Radar_params.R * cos(theta); Radar_params.R * sin(theta)];

% ��Դdoa(��ά):[cos(phi), sin(phi)]
h = [cosd(Src_params.phi); sind(Src_params.phi)];

% ��Ԫ1����������Ԫ���ӳ�
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
        % ʹ��˲ʱƵ�ʼ�����λ��
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

% stft�任
stft_winlen = 128;
win = hann(stft_winlen);
noverlap = stft_winlen / 2;
fft_length = 1024;

X = zeros(Radar_params.RadarAmount, fft_length, floor((Src_params.N - noverlap) / (stft_winlen - noverlap)));
for k = 1:Radar_params.RadarAmount
    [X(k, :, :), f, t1] = stft(x(k, :), 1/Src_params.dt, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', fft_length);
end

window = 2;
noverlap = 1;
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
epsilon = 0.1;  % ���ϵ����ֵ
single_source_regions = find(mean_r > 1 - epsilon);  % ��Դ������ʱƵ����

% ������Դ������ʱƵ����
doa = zeros(length(single_source_regions), 1);  % ֻ��Ҫ�洢phi�Ƕ�
Xi = zeros(1, phi_ed - phi_st + 1);
disp(['��Դ������ʱƵ����������', num2str(length(single_source_regions))]);

for i = 1:length(single_source_regions)
    g = single_source_regions(i);
    st = (g - 1) * (window - noverlap) + 1;
    ed = min(st + window - 1, fft_length);

    g_f = f(st:ed);
    omega = 2 * pi * g_f;

    hat_X = X(:, st:ed, :);
    hat_X = reshape(hat_X, Radar_params.RadarAmount * length(g_f), []);

    V_idx = 0;
    xi = zeros(1, phi_ed - phi_st + 1);
    for ph = phi_st:phi_ed
        h = [cosd(ph); sind(ph)];
        tau = h' * tau_;
        
        hat_A = compute_block_diagonal_matrix(omega, tau, Radar_params.RadarAmount);
        [Q, Lambda] = eig(hat_A);
        
        if V_idx == 0
            V_idx = find(int16(diff(diag(Lambda))) == 1);
        end
        V = Q(:, 1:V_idx);
        xi(ph - phi_st + 1) = xi(ph - phi_st + 1) + 1 / norm(V' * hat_X);
    end
    
    fprintf('\n');
    disp(['��', num2str(i), '����Դ������ʱƵ����', ' ���ϵ����', num2str(mean_r(g)), '����Χ��  ', num2str(st), ' - ', num2str(ed)]);

    [max_val, max_idx] = max(xi);
    doa(i) = max_idx + phi_st - 1;

    disp(['���ֵ��', num2str(max_val), '��phi: ', num2str(doa(i)), '��']);
    Xi = Xi + xi;
end

% doa���ƽ��
[max_val, max_idx] = max(Xi);
t_doa = max_idx + phi_st - 1;

% doa����ͼ
figure;
hold on;
scatter(doa, zeros(size(doa)), '+', 'b');  % DOA ���ƽ������ɫ '+' 
scatter(Src_params.phi, zeros(size(Src_params.phi)), 'o', 'r');  % ��ʵ��Դλ�ã���ɫ 'o' 
scatter(t_doa, 0, 'x', 'g');  % ���� DOA ���ƽ������ɫ 'x'
xlabel('phi (��)');
ylabel('');
legend('DOA ����', '��ʵ��Դλ��', '���� DOA ����');
xlim([phi_st, phi_ed]);
hold off;

% ��ȡÿ��ʱƵ���������Ƶ��
doa_freqs = zeros(length(single_source_regions), 1);
for i = 1:length(single_source_regions)
    g = single_source_regions(i);
    st = (g - 1) * (window - noverlap) + 1;
    ed = min(st + window - 1, fft_length);
    doa_freqs(i) = mean(f(st:ed));  % ʹ�ø�ʱƵ�����ƽ��Ƶ��
end

% ����Ƶ��-DOA����ͼ
figure;
hold on;
% ����DOA���ƽ������ɫ'+'��
scatter(doa_freqs, doa, 100, '+', 'b');
% ������ʵ��Դλ�ã���ɫˮƽ�ߣ�
for i = 1:length(Src_params.phi)
    yline(Src_params.phi(i), 'r--');
end

xlabel('Ƶ�� (Hz)');
ylabel('DOA����ֵ (��)');
title('Ƶ��-DOA���ƽ��');
legend('DOA����', '��ʵ��Դλ��');
grid on;
ylim([phi_st, phi_ed]);
hold off;

function D = compute_block_diagonal_matrix(omega, tau, M)
    % ����:
    % omega: Ƶ������ [omega_1, omega_2, ..., omega_F]
    % tau: ʱ�Ӿ��� tau_{1,k}(phi), ��СΪ 1 x M
    % M: ���� a �ĳ���
    %
    % ���:
    % D: �ֿ�ԽǾ���

    F = length(omega); % Ƶ�ʵ������

    a = exp(1j * omega(:) * tau);

    D = zeros(M * F, M * F);
    for f = 1:F
        a_f = a(f, :).';                                % ��ȡ��ǰƵ�ʵ� a(omega_f, phi)
        D((f-1)*M+1:f*M, (f-1)*M+1:f*M) = (1/M) * (a_f * a_f' - eye(M)); % ���� A_1(omega_f)
    end
end