#!/bin/bash
#SBATCH --job-name=zylia_completo
#SBATCH --partition=a100          # ajustar a la partición GPU de tu clúster
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=zylia-completo-%j.out
#SBATCH --signal=B:TERM@30
#SBATCH --dependency=singleton

set -euo pipefail

# Configuración (se puede sobreescribir con variables de entorno al hacer sbatch)
# Lanzar desde apps/web_monitor/ del repositorio.
SERVICE_DIR="${SERVICE_DIR:-$SLURM_SUBMIT_DIR}"
REPO_DIR="${REPO_DIR:-$(cd "$SERVICE_DIR/../.." && pwd)}"
ENV_DIR="${ENV_DIR:-$REPO_DIR/.venv}"          # entorno de Python con PyTorch y demucs
WEB_HOST="${WEB_HOST:-127.0.0.1}"              # interfaz interna del nodo para el proxy
WEB_PORT="${WEB_PORT:-8088}"
: "${CKPT_PATH:?Define CKPT_PATH con la ruta al checkpoint del modelo}"
IMAGE="$SERVICE_DIR/zylia-monitor-web.sif"
WEB_PID=""
WEB_PGID=""
SERVER_PID=""
SERVER_PGID=""

listener_pids() {
    local port="$1"
    ss -lntpH 2>/dev/null \
        | awk -v suffix=":${port}" '$4 ~ suffix"$" {print}' \
        | grep -o 'pid=[0-9]*' \
        | cut -d= -f2 \
        | sort -u || true
}

port_is_busy() {
    ss -lntH 2>/dev/null | awk -v suffix=":$1" '$4 ~ suffix"$" {found=1} END {exit !found}'
}

terminate_group() {
    local pgid="$1"
    [[ -z "$pgid" ]] && return
    if kill -0 -- "-$pgid" 2>/dev/null; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 -- "-$pgid" 2>/dev/null || return
            sleep 0.25
        done
        kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
}

cleanup() {
    trap - EXIT INT TERM
    echo "[*] Cerrando servicio completo..."
    terminate_group "$SERVER_PGID"
    terminate_group "$WEB_PGID"
    [[ -n "$SERVER_PID" ]] && wait "$SERVER_PID" 2>/dev/null || true
    [[ -n "$WEB_PID" ]] && wait "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ ! -f "$IMAGE" ]]; then
    echo "No existe la imagen: $IMAGE" >&2
    exit 1
fi

# Evitar dos servicios simultáneos incluso si se envía el SBATCH dos veces.
exec 9>"$SERVICE_DIR/zylia-service.lock"
if ! flock -n 9; then
    echo "Ya existe otra instancia de Zylia en ejecución." >&2
    exit 1
fi

# Eliminar únicamente listeners huérfanos pertenecientes a este usuario y a
# esta aplicación. Nunca se termina un proceso ajeno o desconocido.
for port in 5555 7777 "$WEB_PORT"; do
    for pid in $(listener_pids "$port"); do
        [[ -r "/proc/$pid/status" ]] || continue
        owner_uid=$(stat -c '%u' "/proc/$pid")
        command=$(tr '\0' ' ' < "/proc/$pid/cmdline")
        if [[ "$owner_uid" != "$UID" ]]; then
            echo "El puerto $port pertenece a otro usuario (PID $pid)." >&2
            exit 1
        fi
        if [[ "$command" == *"Servidor_web_app.py"* || "$command" == *"uvicorn app:app"* ]]; then
            echo "[!] Cerrando proceso huérfano PID $pid del puerto $port"
            kill -TERM "$pid" 2>/dev/null || true
        else
            echo "El puerto $port está ocupado por un proceso desconocido: $command" >&2
            exit 1
        fi
    done
done

for _ in $(seq 1 30); do
    if ! port_is_busy 5555 && ! port_is_busy 7777 && ! port_is_busy "$WEB_PORT"; then
        break
    fi
    sleep 0.5
done
for port in 5555 7777 "$WEB_PORT"; do
    if port_is_busy "$port"; then
        echo "No se pudo liberar el puerto $port." >&2
        exit 1
    fi
done

# Arrancar la web en un grupo de procesos propio para garantizar que Uvicorn,
# Apptainer y sus hijos se cierren juntos al terminar el trabajo.
echo "[*] Arrancando web en http://$WEB_HOST:$WEB_PORT"
cd "$SERVICE_DIR"
setsid apptainer exec --cleanenv --pwd /app \
    --env STREAM_HOST=127.0.0.1 \
    --env STREAM_PORT=7777 \
    "$IMAGE" \
    python -m uvicorn app:app \
        --host "$WEB_HOST" --port "$WEB_PORT" --no-access-log &
WEB_PID=$!
WEB_PGID=$WEB_PID

for _ in $(seq 1 40); do
    if ! kill -0 -- "-$WEB_PGID" 2>/dev/null; then
        echo "El contenedor web terminó durante el arranque." >&2
        wait "$WEB_PID" || true
        exit 1
    fi
    if curl --fail --silent http://$WEB_HOST:$WEB_PORT/health >/dev/null; then
        echo "[+] Web preparada; entrada procesada en 127.0.0.1:7777"
        break
    fi
    sleep 0.5
done
if ! curl --fail --silent http://$WEB_HOST:$WEB_PORT/health >/dev/null; then
    echo "La web no respondió antes del tiempo límite." >&2
    exit 1
fi

# Cargar el modelo en otro grupo de procesos controlado.
cd "$REPO_DIR"
export MPLCONFIGDIR="$HOME/.config/matplotlib"
export CKPT_PATH
export AUDIO_INPUT_PORT=5555
export WEB_STREAM_HOST=127.0.0.1
export WEB_STREAM_PORT=7777

echo "[*] Cargando Beamformer + HDemucs en la A100..."
setsid "$ENV_DIR/bin/python" -u "$REPO_DIR/apps/network/Servidor_web_app.py" &
SERVER_PID=$!
SERVER_PGID=$SERVER_PID

# Supervisar ambos servicios: si cualquiera cae, finalizar todo el trabajo.
while true; do
    if ! kill -0 -- "-$WEB_PGID" 2>/dev/null; then
        echo "El servicio web se ha detenido inesperadamente." >&2
        exit 1
    fi
    if ! kill -0 -- "-$SERVER_PGID" 2>/dev/null; then
        set +e
        wait "$SERVER_PID"
        status=$?
        set -e
        echo "El servidor GPU ha terminado con código $status." >&2
        exit "$status"
    fi
    sleep 2
done
