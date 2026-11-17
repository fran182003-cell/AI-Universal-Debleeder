function simulated_rir_generator(start_id,nrooms,nworkers)
% SIMULATED_RIR_GENERATOR Generate simulated RIRs using SMIR-generator

addpath(genpath('./src/SMIR-Generator'));

% PARALLEL POOL
if nargin < 3, nworkers = 12; end

% Load parameters
P = readyaml('config2.yaml');
dataset_dir= P.rutas.dataset_dir;

% DIRECTORIES
if ~exist(fullfile(dataset_dir,'rir'),'dir'), mkdir(fullfile(dataset_dir,'rir')); end
if ~exist(fullfile(dataset_dir,'rir_clean'),'dir'), mkdir(fullfile(dataset_dir,'rir_clean')); end
if ~exist(fullfile(dataset_dir,'info'),'dir'), mkdir(fullfile(dataset_dir,'info')); end

% SMIR-generator parameters
c = P.general.c;   % speed of sound m/s
fs = P.general.fs; % processing sampling frequency in Hz 
                                        
N_harm = P.rir_generator.N_harm; 
K = P.rir_generator.K; 
reflectionOrder = P.rir_generator.reflectionOrder; 
src_type = P.rir_generator.src_type; 
src_dist = P.rir_generator.src_dist; 
HP = P.rir_generator.HP;  

% ROOM
room_width = P.rir_generator.room_width;   
room_length = P.rir_generator.room_length; 
room_height = P.rir_generator.room_height; 

% NUEVOS PARÁMETROS: T60 y Fuentes
t60_pool = P.rir_generator.t60_min : P.rir_generator.t60_paso : P.rir_generator.t60_max;
f_min = P.rir_generator.fuentes_min;
f_max = P.rir_generator.fuentes_max;
f_z_pos = P.rir_generator.fuentes_z_pos;
paso_ang = P.rir_generator.paso_angular_deg;

% MICROPHONE ARRAY 
micType = P.rir_generator.micType; 
micR = P.eigenmike.mic_r; 
mic_az = P.eigenmike.mic_az(:); 
mic_el = P.eigenmike.mic_el(:); 
mic_height = P.eigenmike.mic_height;  

%% Tablas para almacenar la información de forma segura para parfor
room_info_cell = cell(nrooms, 5); 
rir_info_cell = cell(nrooms, 1); 

%% GENERATE SIMULATED RIRS
parfor (r_id = 1:nrooms, nworkers)
    rooms = start_id:start_id+nrooms-1;
    room_i = rooms(r_id);

    % Sample room dimensions
    L = [urand(room_width), urand(room_length), urand(room_height)];
    
    % --- COMPROBACIÓN DEL T60 MÍNIMO ---
    V = L(1) * L(2) * L(3);
    S_sup = 2 * (L(1)*L(2) + L(1)*L(3) + L(2)*L(3));
    min_t60 = (24 * V * log(10)) / (c * S_sup);
    
    % Filtrar el pool de T60 para no violar la absorción
    t60_validos = t60_pool(t60_pool >= min_t60);
    if isempty(t60_validos)
        rt60 = min_t60;
    else
        rt60 = t60_validos(randi(length(t60_validos)));
    end
    
    nsample = round(rt60 * fs);
    % Forzamos nsample a par para evitar error en la matriz de SMIR
    if mod(nsample, 2) ~= 0, nsample = nsample + 1; end

    % Display info
    w = getCurrentWorker; n = w.ProcessId;
    fprintf('\nRoom %d:\tL: %.2f %.2f %.2f \trt60: %.2f\t process: %d', room_i, L(1), L(2), L(3), rt60, n);
    
    % Asignación segura a la celda general
    room_info_cell(r_id, :) = {room_i, L(1), L(2), L(3), rt60};
    
    % Microphone position
    micPos = [L(1)/2, L(2)/2, mic_height];
    
    % --- SELECCIÓN ALEATORIA DE FUENTES EN PLANO HORIZONTAL ---
    num_fuentes = randi([f_min, f_max]);
    
    % 1. Ángulo de inicio aleatorio entre 0.0 y 359.9 (con 1 decimal)
    ang_start = round(rand() * 359.9, 1);
    
    % 2. Generar las posiciones posibles (tantas como fuentes_max) espaciadas por paso_ang
    % Si f_max es 7 y paso_ang es 50, genera 7 posiciones (ej: 0, 50, 100, 150, 200, 250, 300)
    % sumadas al ángulo de inicio.
    posiciones_totales = ang_start + (0 : f_max - 1) * paso_ang;
    
    % 3. Aplicar módulo 360 para que las posiciones que se pasan den la vuelta
    % (ejemplo: 340 + 50 = 390 -> se convierte en 30 grados)
    azimuths_disponibles = mod(posiciones_totales, 360);
    
    % 4. Seleccionar aleatoriamente 'num_fuentes' de esas 7 posiciones disponibles
    idx_fuentes = randperm(length(azimuths_disponibles), num_fuentes);
    azimuths_sel = azimuths_disponibles(idx_fuentes);
    
    temp_rir_info = cell(num_fuentes, 7);
    
    % For each source, generate RIR
    for rir_cnt = 1:num_fuentes
        az_tmp = deg2rad(azimuths_sel(rir_cnt));
        
        inside = false;
        intentos = 0;
        while not(inside) && intentos < 50
            intentos = intentos + 1;
            r_tmp = urand(src_dist);
            
            % Forzamos la altura Z absoluta a f_z_pos
            src_x = micPos(1) + r_tmp * cos(az_tmp);
            src_y = micPos(2) + r_tmp * sin(az_tmp);
            src_z = f_z_pos; 
            src = [src_x, src_y, src_z];
            
            % Verificación inline
            inside = (src(1)>=0 && src(1)<=L(1) && src(2)>=0 && src(2)<=L(2) && src(3)>=0 && src(3)<=L(3));
        end
        
        if ~inside
            fprintf('\n\tAviso: Fuente %d no cabe en la sala tras 50 intentos.', rir_cnt);
            continue;
        end

        % Calcular elevación relativa al micrófono
        [xtmp, ytmp, ztmp] = mysph2cart(az_tmp, pi/2, r_tmp); % Guardamos pi/2 xq esta en plano 2D relativo a su distancia
        
        temp_rir_info(rir_cnt, :) = {room_i, rir_cnt, src_x, src_y, src_z, az_tmp, pi/2};
        
        [a, b] = mycart2sph(micPos(1)-src(1), micPos(2)-src(2), micPos(3)-src(3));
        src_ang = [a, b];
        
        rir_name = sprintf('room%d_%d', room_i, rir_cnt);
        fprintf('\n\t%d: %s', rir_cnt, rir_name);
        
        % SMIR-generator
        [rir, ~] = smir_generator(c, fs, micPos, src, L, rt60, micType, micR, [mic_az,mic_el], N_harm, nsample, K, reflectionOrder, 0, HP, src_type, src_ang);
        % Usamos min_t60 para el ground truth clean
    

        % Guardado directo mediante matfile para evitar error de transparencia
        archivo_rir = fullfile(dataset_dir, 'rir', sprintf('%s.mat', rir_name));
        m_rir = matfile(archivo_rir, 'Writable', true);
        m_rir.sim_rir = rir;
        
       
    end
    
    rir_info_cell{r_id} = temp_rir_info;
end

% SAVE ROOM AND RIR INFO
rir_info_mat = vertcat(rir_info_cell{:});
empty_rows = cellfun(@isempty, rir_info_mat(:,1));
rir_info_mat(empty_rows, :) = [];

room_info = cell2table(room_info_cell, 'VariableNames', {'room_id', 'x', 'y', 'z','t60'});
rir_info = cell2table(rir_info_mat, 'VariableNames', {'room_id','rir_id', 'x', 'y', 'z','az','el'});

writetable(room_info, fullfile(dataset_dir,'info',sprintf('room_info_%d_%d.csv',start_id,start_id+nrooms-1)));
writetable(rir_info, fullfile(dataset_dir,'info',sprintf('rir_info_%d_%d.csv',start_id,start_id+nrooms-1)));

end

%% HELPER FUNCTIONS
function [x] = urand(interval)
    x = rand()*(interval(2)- interval(1))+interval(1);
end