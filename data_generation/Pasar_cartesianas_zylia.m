% Matriz con las coordenadas [x, y, z] del ZYLIA (extraída del PDF)
xyz = [
    0.0000,  0.0000,  0.0490;
    0.0327,  0.0010,  0.0365;
   -0.0164,  0.0283,  0.0365;
   -0.0163, -0.0283,  0.0365;
    0.0064, -0.0458,  0.0163;
    0.0366, -0.0282,  0.0163;
    0.0365,  0.0284,  0.0163;
    0.0062,  0.0458,  0.0163;
   -0.0428,  0.0174,  0.0163;
   -0.0427, -0.0176,  0.0163;
   -0.0365, -0.0284, -0.0163;
   -0.0062, -0.0458, -0.0163;
    0.0428, -0.0174, -0.0163;
    0.0427,  0.0176, -0.0163;
   -0.0063,  0.0458, -0.0163;
   -0.0366,  0.0282, -0.0163;
   -0.0327,  0.0010, -0.0365;
    0.0164, -0.0283, -0.0365;
    0.0163,  0.0283, -0.0365
];

% Cálculo de r, azimut y ángulo de inclinación (polar)
r = sqrt(xyz(:,1).^2 + xyz(:,2).^2 + xyz(:,3).^2);
az = atan2(xyz(:,2), xyz(:,1));
inc = acos(xyz(:,3) ./ r);

% Ajustar azimut para que esté en el rango [0, 2*pi]
az(az < 0) = az(az < 0) + 2*pi;

% Imprimir en formato YAML
fprintf('\n--- COPIA ESTO EN TU CONFIG.YAML ---\n');
fprintf('zylia:\n');
fprintf('  micType: ''rigid''\n');
fprintf('  mic_height: 1.2\n');
fprintf('  mic_r: 0.0490\n');
fprintf('  mic_az: [%s]\n', join(string(round(az, 4)), ', '));
fprintf('  mic_el: [%s]\n', join(string(round(inc, 4)), ', ')); % SMIR usa inclinación aquí
fprintf('  gain: [%s]\n', join(string(ones(19,1)), ', '));