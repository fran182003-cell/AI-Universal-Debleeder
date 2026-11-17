% --- CONFIGURACIÓN DE LA PRUEBA ---
id_inicial = 1;     % Empezamos enumerando desde la "Sala 1"
num_salas = 4;      % Generamos solo 1 sala para probar rápido
num_workers = 0;    % IMPORTANTE: Ponemos 0 para ver errores en pantalla

% --- EJECUCIÓN ---
simulated_rir_generator(id_inicial, num_salas, num_workers);