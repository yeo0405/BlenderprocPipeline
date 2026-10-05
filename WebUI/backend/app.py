"""
Flask backend API for the 6D Object Detection Training Pipeline UI.
Handles model import and training data generation.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import threading
import queue
import shutil
import re
import pty
import select
import shlex
import errno
import secrets
import hmac
from datetime import datetime
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max file size

# Disable Flask request logging to show only training information
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Only show errors, not info/debug requests

# Fixed BOP parent path (do not expose editable in UI)
BACKEND_DIR = Path(__file__).resolve().parent
WEBUI_DIR = BACKEND_DIR.parent
AUTOMATED_DIR = WEBUI_DIR.parent
BOP_PARENT_PATH = str((AUTOMATED_DIR / 'bop' / 'BlenderProc').resolve())
AUTOMATED_VENV_BIN = str((AUTOMATED_DIR / '.venv' / 'bin').resolve())
BLENDERPROC_EXECUTABLE = str((Path(AUTOMATED_VENV_BIN) / 'blenderproc').resolve())
BLENDERPROC_PYTHON = os.environ.get('DATA_GEN_PYTHON') or str((Path(AUTOMATED_VENV_BIN) / 'python').resolve())
DEFAULT_OUTPUT_DIR = str((AUTOMATED_DIR / 'output').resolve())
DEFAULT_OUTPUT_DATA_DIR = str((Path(DEFAULT_OUTPUT_DIR) / 'dataset').resolve())
TERMINAL_JOB_STATE_DIR = str((BACKEND_DIR / '.terminal_job_state').resolve())
YOLO_TRANSFER_SCRIPT = str((AUTOMATED_DIR / 'utils' / 'yolo_transfer.py').resolve())
YOLO_CUSTOM_CONFIG_PATH = str((AUTOMATED_DIR / 'output' / 'dataset' / 'data.yaml').resolve())
GENERATE_CALLBACK_URL = 'http://127.0.0.1:5000/api/generate/callback'
DEFAULT_TRAIN_SPLIT_RATIO = 0.8
MIRROR_BLENDERPROC_STDOUT = os.environ.get('WEBUI_MIRROR_BLENDERPROC_STDOUT', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
# Relative path to venv activate script (from AUTOMATED_DIR)
VENV_ACTIVATE_RELATIVE = '.venv/bin/activate'

# Global state for tracking long-running processes
job_queue = queue.Queue()
job_results = {}
current_job = None

# File upload configuration
ALLOWED_EXTENSIONS = {'blend'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def infer_completed_scenes_from_output(job_status: Dict) -> int:
    """Infer completed scenes from generated files as a fallback to log parsing."""
    try:
        params = job_status.get('parameters') or {}
        output_dir = params.get('output_dir') or job_status.get('output_dir') or DEFAULT_OUTPUT_DIR
        sample_size = int(params.get('sample_size') or 0)
        total_scenes = int(job_status.get('total_scenes') or 0)
        if not output_dir or total_scenes <= 0:
            return 0

        dataset_name = job_status.get('effective_dataset_name') or 'hb'
        train_pbr_dir = os.path.join(
            output_dir,
            'bop_data',
            dataset_name,
            'train_pbr'
        )

        if not os.path.isdir(train_pbr_dir):
            return 0

        scene_dirs: List[str] = []
        for entry in os.listdir(train_pbr_dir):
            scene_dir = os.path.join(train_pbr_dir, entry)
            if os.path.isdir(scene_dir) and re.fullmatch(r'\d{6}', entry):
                scene_dirs.append(scene_dir)

        completed_by_scene_dirs = 0
        if scene_dirs:
            required_scene_files = ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json')
            for scene_dir in scene_dirs:
                if all(os.path.isfile(os.path.join(scene_dir, filename)) for filename in required_scene_files):
                    completed_by_scene_dirs += 1

        # If no scene files found but have completed scene dirs, return that count
        if completed_by_scene_dirs > 0:
            return max(0, min(total_scenes, completed_by_scene_dirs))

        # Fallback: count completed scenes by mask images
        if sample_size <= 0:
            # If sample_size not provided, default to 100
            sample_size = 100

        completed_by_masks = 0
        for scene_dir in scene_dirs if scene_dirs else [os.path.join(train_pbr_dir, '000000')]:
            mask_visib_dir = os.path.join(scene_dir, 'mask_visib')
            if os.path.isdir(mask_visib_dir):
                generated_images = len([f for f in os.listdir(mask_visib_dir) if f.endswith('.png')])
                if sample_size > 0:
                    completed_by_masks += generated_images // sample_size
                # If we have any images but sample_size is invalid, count the scene as completed
                if generated_images > 0 and sample_size <= 0:
                    completed_by_masks += 1

        inferred_scenes = max(completed_by_scene_dirs, completed_by_masks)
        return max(0, min(total_scenes, inferred_scenes))
    except Exception as e:
        logger.warning(f"Failed to infer scenes from output: {str(e)}")
        return 0


def parse_exit_code_from_file(exit_status_file: Optional[str]) -> Optional[int]:
    if not exit_status_file or not os.path.exists(exit_status_file):
        return None
    try:
        with open(exit_status_file, 'r', encoding='utf-8') as f:
            exit_code_text = f.read().strip()
        if not exit_code_text:
            return None
        return int(exit_code_text)
    except Exception as e:
        logger.warning(f"Failed to read exit code file '{exit_status_file}': {str(e)}")
        return None


def parse_exit_code_from_log_text(log_text: str) -> Optional[int]:
    if not log_text:
        return None
    matches = re.findall(r'Generation finished with exit code:\s*(-?\d+)', log_text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def is_generation_output_complete(job_status: Dict) -> Tuple[bool, int, int]:
    try:
        total_scenes = int(job_status.get('total_scenes') or 0)
        if total_scenes <= 0:
            return False, 0, 0

        params = job_status.get('parameters') or {}
        output_dir = params.get('output_dir') or job_status.get('output_dir') or DEFAULT_OUTPUT_DIR
        dataset_name = job_status.get('effective_dataset_name') or 'hb'
        train_pbr_dir = os.path.join(output_dir, 'bop_data', dataset_name, 'train_pbr')
        if not os.path.isdir(train_pbr_dir):
            return False, 0, total_scenes

        completed_scene_dirs = 0
        required_scene_files = ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json')

        for entry in os.listdir(train_pbr_dir):
            if not re.fullmatch(r'\d{6}', entry):
                continue
            scene_dir = os.path.join(train_pbr_dir, entry)
            if not os.path.isdir(scene_dir):
                continue
            if all(os.path.isfile(os.path.join(scene_dir, filename)) for filename in required_scene_files):
                completed_scene_dirs += 1

        return completed_scene_dirs >= total_scenes, completed_scene_dirs, total_scenes
    except Exception as e:
        logger.warning(f"Failed to validate generation output completeness: {str(e)}")
        return False, 0, 0


def is_pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def ensure_terminal_job_state_dir() -> None:
    os.makedirs(TERMINAL_JOB_STATE_DIR, exist_ok=True)


def truncate_to_six_decimals(value: float) -> float:
    return int(float(value) * 1_000_000) / 1_000_000


def load_train_test_split_runner():
    module_path = AUTOMATED_DIR / 'utils' / 'train_test_split.py'
    spec = importlib.util.spec_from_file_location('automated_output_train_test_split', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load train_test_split module from {module_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'run_split'):
        raise AttributeError(f'Module {module_path} does not define run_split')

    return module.run_split


def find_terminal_emulator() -> Optional[Tuple[str, str]]:
    candidates = [
        ('terminator', 'terminator'),
        ('gnome-terminal', 'gnome-terminal'),
        ('x-terminal-emulator', 'x-terminal-emulator'),
        ('xterm', 'xterm'),
        ('konsole', 'konsole'),
        ('xfce4-terminal', 'xfce4-terminal'),
    ]
    for terminal_name, executable in candidates:
        resolved = shutil.which(executable)
        if resolved:
            return terminal_name, resolved
    return None


def build_terminal_launch_command(terminal_name: str, executable: str, shell_command: str) -> List[str]:
    if terminal_name == 'terminator':
        return [executable, '-x', 'bash', '-lc', shell_command]
    if terminal_name == 'gnome-terminal':
        return [executable, '--', 'bash', '-lc', shell_command]
    if terminal_name == 'x-terminal-emulator':
        return [executable, '-e', 'bash', '-lc', shell_command]
    if terminal_name == 'xterm':
        return [executable, '-hold', '-e', 'bash', '-lc', shell_command]
    if terminal_name == 'konsole':
        return [executable, '-e', 'bash', '-lc', shell_command]
    if terminal_name == 'xfce4-terminal':
        return [executable, '--hold', '--command', f'bash -lc {shlex.quote(shell_command)}']
    raise ValueError(f'Unsupported terminal emulator: {terminal_name}')


def launch_generation_in_system_terminal(job_id: str, command: List[str], work_dir: str, env: Dict[str, str], stdin_text: Optional[str] = None) -> Tuple[subprocess.Popen, str, str, str]:
    ensure_terminal_job_state_dir()

    if not env.get('DISPLAY'):
        raise RuntimeError('DISPLAY is not set. A graphical system terminal cannot be opened from the backend.')

    terminal_info = find_terminal_emulator()
    if terminal_info is None:
        raise RuntimeError('No supported terminal emulator found. Install terminator, gnome-terminal, x-terminal-emulator, xterm, konsole, or xfce4-terminal.')

    terminal_name, terminal_executable = terminal_info
    exit_status_file = os.path.join(TERMINAL_JOB_STATE_DIR, f'{job_id}.exit_code')
    log_file = ''
    command_str = shlex.join(command)
    run_command_str = command_str
    if stdin_text is not None:
        run_command_str = f'printf %b {shlex.quote(stdin_text)} | {command_str}'
    # Source venv activation from Automated directory (using relative path)
    automated_dir_str = str(AUTOMATED_DIR)
    venv_activate_path = os.path.join(automated_dir_str, VENV_ACTIVATE_RELATIVE)
    shell_command = (
        f'cd {shlex.quote(automated_dir_str)} && '
        f'source {shlex.quote(venv_activate_path)} && '
        f'cd {shlex.quote(work_dir)} && '
        'export TERM=${TERM:-xterm-256color}; '
        'export PYTHONUNBUFFERED=1; '
        f'{run_command_str}; '
        'exit_code=$?; '
        f'printf "%s" "$exit_code" > {shlex.quote(exit_status_file)}; '
        'echo; '
        'echo "Generation finished with exit code: $exit_code"; '
        'echo "Press Enter to close this terminal..."; '
        'read'
    )
    terminal_command = build_terminal_launch_command(terminal_name, terminal_executable, shell_command)
    terminal_process = subprocess.Popen(
        terminal_command,
        cwd=work_dir,
        env=env,
        start_new_session=True,
    )
    return terminal_process, exit_status_file, terminal_name, log_file


def infer_completed_scenes_from_terminal_log(job_status: Dict) -> Tuple[int, str]:
    """Infer completed scenes by parsing system terminal log output."""
    log_file = job_status.get('terminal_log_file')
    if not log_file or not os.path.isfile(log_file):
        return 0, ''

    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            log_text = f.read()
    except Exception as e:
        logger.warning(f"Failed to read terminal log: {str(e)}")
        return 0, ''

    log_text = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', log_text)
    completed_rounds = 0
    for line in log_text.splitlines():
        lower_line = line.lower()
        if 'calculating coco annotations' in lower_line:
            completed_rounds += 1

    return completed_rounds, log_text[-30000:]


def read_terminal_log_tail(job_status: Dict, max_chars: int = 30000) -> str:
    log_file = job_status.get('terminal_log_file')
    if not log_file or not os.path.isfile(log_file):
        return ''

    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            log_text = f.read()
        log_text = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', log_text)
        return log_text[-max_chars:]
    except Exception as e:
        logger.warning(f"Failed to read terminal log tail: {str(e)}")
        return ''


class DatasetValidator:
    """Validate dataset files and structure."""
    
    REQUIRED_FILES = {
        'camera': 'camera.json',
        'targets': 'test_targets_bop19.json',
    }
    
    @staticmethod
    def validate_dataset(dataset_path: str, dataset_name: str) -> Tuple[bool, Dict[str, str]]:
        """
        Validate that a dataset has all required files.
        
        Returns:
            Tuple of (is_valid, errors_dict)
        """
        errors = {}
        base_path = os.path.join(dataset_path, dataset_name)
        
        if not os.path.exists(base_path):
            return False, {f"dataset_path": f"Dataset path not found: {base_path}"}
        
        for file_type, file_name in DatasetValidator.REQUIRED_FILES.items():
            file_path = os.path.join(base_path, file_name)
            if not os.path.exists(file_path):
                errors[file_type] = f"Missing {file_name}: {file_path}"
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_models(dataset_path: str, dataset_name: str) -> Tuple[List[str], List[str]]:
        """
        List available .blend files and validate them.
        
        Returns:
            Tuple of (models_list, errors_list)
        """
        models_dir = os.path.join(dataset_path, dataset_name, 'models')
        models = []
        errors = []
        
        if not os.path.exists(models_dir):
            errors.append(f"Models directory not found: {models_dir}")
            return models, errors
        
        try:
            for file in os.listdir(models_dir):
                if file.endswith('.blend'):
                    models.append(file)
        except Exception as e:
            errors.append(f"Error reading models directory: {str(e)}")
        
        return models, errors
    
    @staticmethod
    def load_config_file(dataset_path: str, dataset_name: str, 
                        config_file: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Load and parse a JSON configuration file."""
        file_path = os.path.join(dataset_path, dataset_name, config_file)
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f), None
        except FileNotFoundError:
            return None, f"File not found: {file_path}"
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON in {config_file}: {str(e)}"
        except Exception as e:
            return None, f"Error reading {config_file}: {str(e)}"


def run_blenderproc_job(job_id: str, command: List[str], work_dir: str, env: Dict[str, str]):
    """Execute BlenderProc command in a separate thread."""
    try:
        logger.info(f"[Job {job_id}] Starting in {work_dir}: {' '.join(command)}")

        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            command,
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=subprocess.DEVNULL,
            cwd=work_dir,
            env=env,
            bufsize=0,
            text=False
        )
        os.close(slave_fd)

        output_text = ""
        max_output_chars = 300000
        scene_id_pattern = re.compile(r"train_pbr/(\d{6})")
        completed_scene_count = 0

        job_entry = job_results.get(job_id, {})
        total_scenes = int(job_entry.get('total_scenes') or 0)

        def push_running_state() -> None:
            completed_scenes = completed_scene_count
            progress_percent = 0
            if total_scenes > 0:
                progress_percent = int((completed_scenes / total_scenes) * 100)

            existing = job_results.get(job_id, {})
            existing.update({
                'status': 'running',
                'stdout': output_text,
                'stderr': '',
                'completed_scenes': completed_scenes,
                'progress_percent': progress_percent,
            })
            job_results[job_id] = existing

        def mirror_console_line(line_text: str) -> None:
            if not MIRROR_BLENDERPROC_STDOUT:
                return
            clean_line = line_text.strip()
            if not clean_line:
                return
            logger.info(f"[BlenderProc][{job_id}] {clean_line}")

        current_line = ""
        try:
            while True:
                if process.poll() is not None:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if not ready:
                        break

                ready, _, _ = select.select([master_fd], [], [], 0.2)
                if not ready:
                    continue

                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as read_error:
                    # PTY returns EIO on EOF after child exits; treat as normal termination.
                    if read_error.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break

                text_chunk = chunk.decode('utf-8', errors='replace')
                for ch in text_chunk:
                    if ch in ('\n', '\r'):
                        if current_line:
                            line = current_line
                            output_text += line + "\n"
                            mirror_console_line(line)
                            if "Calculating COCO annotations" in line:
                                match = scene_id_pattern.search(line)
                                if match:
                                    completed_scene_count += 1
                            current_line = ""
                            if len(output_text) > max_output_chars:
                                output_text = output_text[-max_output_chars:]
                            push_running_state()
                        continue

                    current_line += ch

                    if len(current_line) >= 220:
                        output_text += current_line + "\n"
                        mirror_console_line(current_line)
                        if len(output_text) > max_output_chars:
                            output_text = output_text[-max_output_chars:]
                        current_line = ""
                        push_running_state()
        finally:
            if current_line:
                output_text += current_line + "\n"
                mirror_console_line(current_line)
            if len(output_text) > max_output_chars:
                output_text = output_text[-max_output_chars:]
            push_running_state()
            try:
                os.close(master_fd)
            except OSError:
                pass

        return_code = process.wait(timeout=3600)
        error_message = None
        if return_code != 0:
            error_message = (output_text or "BlenderProc execution failed").strip()[:800]

        existing = job_results.get(job_id, {})
        existing.update({
            'status': 'completed' if return_code == 0 else 'failed',
            'return_code': return_code,
            'stdout': output_text,
            'stderr': '',
            'error': error_message,
            'completed_scenes': completed_scene_count if total_scenes > 0 else existing.get('completed_scenes', 0),
            'progress_percent': 100 if return_code == 0 else existing.get('progress_percent', 0),
            'completed_at': datetime.now().isoformat()
        })
        job_results[job_id] = existing

        logger.info(f"[Job {job_id}] Completed with code {return_code}")

    except subprocess.TimeoutExpired:
        logger.error(f"[Job {job_id}] Timeout after 1 hour")
        existing = job_results.get(job_id, {})
        existing.update({
            'status': 'failed',
            'error': 'Process timeout (>1 hour)',
            'completed_at': datetime.now().isoformat()
        })
        job_results[job_id] = existing
    except Exception as e:
        logger.error(f"[Job {job_id}] Error: {str(e)}")
        existing = job_results.get(job_id, {})
        existing.update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        })
        job_results[job_id] = existing


# API Routes

@app.route('/api/datasets/list', methods=['GET'])
def list_datasets():
    """List all available datasets."""
    try:
        bop_path = request.args.get('bop_path', BOP_PARENT_PATH)
        
        if not os.path.exists(bop_path):
            return jsonify({'error': f'BOP path not found: {bop_path}'}), 404
        
        datasets = []
        for item in os.listdir(bop_path):
            item_path = os.path.join(bop_path, item)
            if os.path.isdir(item_path):
                datasets.append(item)
        
        datasets.sort()
        return jsonify({'datasets': datasets}), 200
        
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/create', methods=['POST'])
def create_dataset():
    """Create a new dataset folder structure."""
    try:
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        bop_path = data.get('bop_path')
        
        if not dataset_name or not bop_path:
            return jsonify({'success': False, 'error': 'Missing dataset_name or bop_path'}), 400
        
        # Sanitize dataset name
        dataset_name = secure_filename(dataset_name)
        
        # Create dataset directory path
        dataset_path = os.path.join(bop_path, dataset_name)
        
        # Check if dataset already exists
        if os.path.exists(dataset_path):
            return jsonify({
                'success': False, 
                'error': f'Dataset "{dataset_name}" already exists at {dataset_path}'
            }), 409
        
        # Create directory structure
        try:
            os.makedirs(dataset_path, exist_ok=True)
            os.makedirs(os.path.join(dataset_path, 'models'), exist_ok=True)
            
            logger.info(f"Created dataset folder: {dataset_path}")
            
            return jsonify({
                'success': True,
                'message': f'Dataset "{dataset_name}" created successfully',
                'dataset_path': dataset_path,
                'models_path': os.path.join(dataset_path, 'models')
            }), 201
            
        except OSError as e:
            return jsonify({
                'success': False,
                'error': f'Failed to create dataset folder: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset_files():
    """Upload blend model files to a dataset."""
    try:
        dataset_name = request.form.get('dataset_name')

        if not dataset_name:
            return jsonify({
                'success': False,
                'error': 'Missing dataset_name'
            }), 400

        dataset_path = os.path.join(BOP_PARENT_PATH, dataset_name)

        if not os.path.exists(dataset_path):
            return jsonify({
                'success': False,
                'error': f'Dataset path not found: {dataset_path}. Create dataset first or use the create_and_upload endpoint.'
            }), 404
        
        uploaded_files = {'blend': []}
        errors = []

        # Handle blend files
        blend_files = request.files.getlist('blend_files')
        models_path = os.path.join(dataset_path, 'models')
        os.makedirs(models_path, exist_ok=True)

        for file in blend_files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(models_path, filename)
                    file.save(file_path)
                    uploaded_files['blend'].append(filename)
                    logger.info(f"Uploaded blend file: {filename} to {models_path}")
                except Exception as e:
                    errors.append(f"Failed to upload {file.filename}: {str(e)}")
        
        response = {
            'success': True,
            'message': 'Blend files uploaded successfully',
            'uploaded_files': uploaded_files,
            'dataset_path': dataset_path
        }
        
        if errors:
            response['warnings'] = errors
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/datasets/create_and_upload', methods=['POST'])
def create_and_upload_dataset():
    """
    Create dataset and upload blend model files in one step.
    Automatically copies camera.json and test_targets_bop19.json from temp folder.
    Overwrites dataset if it already exists.

    Expected multipart form data:
    - dataset_name: Name of the dataset
    - blend_files: Blend model files (one or more)
    """
    try:
        dataset_name = request.form.get('dataset_name') or 'hb'

        # Check if blend files are provided
        blend_files = request.files.getlist('blend_files')
        if not blend_files or not any(f.filename for f in blend_files):
            return jsonify({
                'success': False,
                'error': 'Please upload at least one .blend model file'
            }), 400
        
        # Sanitize dataset name
        dataset_name = secure_filename(dataset_name)
        dataset_path = os.path.join(BOP_PARENT_PATH, dataset_name)
        models_path = os.path.join(dataset_path, 'models')
        
        # Remove existing dataset if it exists (overwrite)
        if os.path.exists(dataset_path):
            try:
                shutil.rmtree(dataset_path)
                logger.info(f"Removed existing dataset folder: {dataset_path}")
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Failed to overwrite existing dataset: {str(e)}'
                }), 500
        
        # Create directory structure
        try:
            os.makedirs(models_path, exist_ok=True)
            logger.info(f"Created dataset folder: {dataset_path}")
        except OSError as e:
            return jsonify({
                'success': False,
                'error': f'Failed to create dataset folder: {str(e)}'
            }), 500
        
        uploaded_files = {'blend': []}
        errors = []
        
        # Handle blend files
        for file in blend_files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(models_path, filename)
                    file.save(file_path)
                    uploaded_files['blend'].append(filename)
                    logger.info(f"Uploaded blend file: {filename} to {models_path}")
                except Exception as e:
                    errors.append(f"Failed to upload {file.filename}: {str(e)}")
        
        # Copy camera.json from temp folder
        temp_camera_path = os.path.join(BOP_PARENT_PATH, 'temp', 'camera.json')
        camera_path = os.path.join(dataset_path, 'camera_primesense.json')
        try:
            if os.path.exists(temp_camera_path):
                shutil.copy2(temp_camera_path, camera_path)
                logger.info(f"Copied camera.json from temp: {camera_path}")
            else:
                # Fallback: create default if temp file doesn't exist
                default_camera = {
                    "fx": 572.4114,
                    "fy": 572.4114,
                    "cx": 325.2611,
                    "cy": 242.0449,
                    "width": 640,
                    "height": 480,
                    "depth_scale": 0.001
                }
                with open(camera_path, 'w') as f:
                    json.dump(default_camera, f, indent=2)
                logger.info(f"Created default camera.json: {camera_path}")
        except Exception as e:
            errors.append(f"Failed to copy/create camera.json: {str(e)}")
        
        # Copy test_targets_bop19.json from temp folder
        temp_targets_path = os.path.join(BOP_PARENT_PATH, 'temp', 'test_targets_bop19.json')
        targets_path = os.path.join(dataset_path, 'test_targets_bop19.json')
        try:
            if os.path.exists(temp_targets_path):
                shutil.copy2(temp_targets_path, targets_path)
                logger.info(f"Copied test_targets_bop19.json from temp: {targets_path}")
            else:
                # Fallback: create default if temp file doesn't exist
                default_targets = [
                    {
                        "obj_id": 1,
                        "inst_count": 1
                    }
                ]
                with open(targets_path, 'w') as f:
                    json.dump(default_targets, f, indent=2)
                logger.info(f"Created default test_targets_bop19.json: {targets_path}")
        except Exception as e:
            errors.append(f"Failed to copy/create test_targets_bop19.json: {str(e)}")
        
        response = {
            'success': True,
            'message': 'Dataset created and blend files uploaded successfully',
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'runtime_dataset_name': 'hb',
            'runtime_dataset_path': dataset_path,
            'uploaded_files': uploaded_files,
            'auto_created': ['camera.json', 'test_targets_bop19.json']
        }
        
        if errors:
            response['warnings'] = errors
        
        return jsonify(response), 201
        
    except Exception as e:
        logger.error(f"Error creating and uploading dataset: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()}), 200


@app.route('/api/dataset/validate', methods=['POST'])
def validate_dataset():
    """
    Validate a dataset has all required files.
    
    Expected JSON:
    {
        "dataset_path": "/path/to/bop",
        "dataset_name": "hb"
    }
    """
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path')
        dataset_name = data.get('dataset_name')
        
        if not dataset_path or not dataset_name:
            return jsonify({
                'valid': False,
                'errors': {'input': 'Missing dataset_path or dataset_name'}
            }), 400
        
        is_valid, errors = DatasetValidator.validate_dataset(dataset_path, dataset_name)
        
        response = {
            'valid': is_valid,
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
        }
        
        if errors:
            response['errors'] = errors
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        return jsonify({
            'valid': False,
            'errors': {'server': str(e)}
        }), 500


@app.route('/api/dataset/models', methods=['POST'])
def get_models():
    """
    Get list of available models in a dataset.
    
    Expected JSON:
    {
        "dataset_path": "/path/to/bop",
        "dataset_name": "hb"
    }
    """
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path')
        dataset_name = data.get('dataset_name')
        
        if not dataset_path or not dataset_name:
            return jsonify({'error': 'Missing parameters'}), 400
        
        models, errors = DatasetValidator.validate_models(dataset_path, dataset_name)
        
        response = {
            'dataset_name': dataset_name,
            'models': models,
            'count': len(models)
        }
        
        if errors:
            response['warnings'] = errors
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/camera', methods=['POST'])
def get_camera_config():
    """
    Get camera intrinsics from dataset.
    
    Expected JSON:
    {
        "dataset_path": "/path/to/bop",
        "dataset_name": "hb"
    }
    """
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path')
        dataset_name = data.get('dataset_name')
        
        if not dataset_path or not dataset_name:
            return jsonify({'error': 'Missing parameters'}), 400
        
        config, error = DatasetValidator.load_config_file(
            dataset_path, dataset_name, 'camera.json'
        )
        
        if error:
            return jsonify({'error': error}), 404
        
        return jsonify({
            'dataset_name': dataset_name,
            'camera_config': config
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting camera config: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/targets', methods=['POST'])
def get_target_config():
    """
    Get target configuration from dataset.
    
    Expected JSON:
    {
        "dataset_path": "/path/to/bop",
        "dataset_name": "hb"
    }
    """
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path')
        dataset_name = data.get('dataset_name')
        
        if not dataset_path or not dataset_name:
            return jsonify({'error': 'Missing parameters'}), 400
        
        config, error = DatasetValidator.load_config_file(
            dataset_path, dataset_name, 'test_targets_bop19.json'
        )
        
        if error:
            return jsonify({'error': error}), 404
        
        # If config is a list, return count and first few items
        if isinstance(config, list):
            return jsonify({
                'dataset_name': dataset_name,
                'target_count': len(config),
                'targets_sample': config[:5],
                'total_targets': len(config)
            }), 200
        
        return jsonify({
            'dataset_name': dataset_name,
            'targets': config
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting target config: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset/models_info', methods=['POST'])
def get_models_info():
    """
    Get blend model files from dataset.
    
    Expected JSON:
    {
        "dataset_path": "/path/to/bop",
        "dataset_name": "hb"
    }
    """
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path')
        dataset_name = data.get('dataset_name')
        
        if not dataset_path or not dataset_name:
            return jsonify({'error': 'Missing parameters'}), 400
        
        models, errors = DatasetValidator.validate_models(dataset_path, dataset_name)

        response = {
            'dataset_name': dataset_name,
            'model_files': models,
            'count': len(models)
        }

        if errors:
            response['warnings'] = errors

        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/start', methods=['POST'])
def start_generation():
    """
    Start training data generation job.
    
    Expected JSON:
    {
        "bop_parent_path": "/path/to/bop",
        "dataset_name": "hb",
        "cc_textures_path": "/path/to/textures",
        "num_scenes": 2000,
        "sample_size": 100,
        "max_samples": 50,
        "seed": null
    }
    """
    try:
        global current_job
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['bop_parent_path']
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400
        
        # Check if a job is already running
        if current_job and current_job not in job_results:
            return jsonify({
                'error': 'A generation job is already running',
                'current_job_id': current_job
            }), 409
        
        # Create job
        job_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        current_job = job_id
        
        # Always run with fixed BOP path and its virtual environment
        bop_parent_path = BOP_PARENT_PATH
        work_dir = bop_parent_path

        # Prefer venv python -m blenderproc to avoid broken shebang in entrypoint scripts.
        use_venv_python_module = os.path.isfile(BLENDERPROC_PYTHON)
        blenderproc_executable = BLENDERPROC_EXECUTABLE if os.path.exists(BLENDERPROC_EXECUTABLE) else 'blenderproc'

        # Use absolute script path to avoid cwd-dependent resolution
        blenderproc_script = os.path.join(work_dir, 'examples', 'datasets', 'bop_challenge', 'main_ui.py')
        script_abs_path = blenderproc_script
        
        # Check if script exists
        if not os.path.exists(script_abs_path):
            return jsonify({
                'error': f'BlenderProc script not found: {script_abs_path}'
            }), 400

        requested_dataset_name = secure_filename(str(data.get('dataset_name', 'hb')).strip()) or 'hb'
        fixed_dataset_name = 'hb'
        dataset_abs_path = os.path.join(bop_parent_path, fixed_dataset_name)

        if not os.path.isdir(dataset_abs_path):
            fallback_dataset_path = os.path.join(bop_parent_path, requested_dataset_name) if requested_dataset_name else None
            if (
                fallback_dataset_path
                and requested_dataset_name != fixed_dataset_name
                and os.path.isdir(fallback_dataset_path)
            ):
                try:
                    if os.path.exists(dataset_abs_path):
                        shutil.rmtree(dataset_abs_path)
                    shutil.copytree(fallback_dataset_path, dataset_abs_path)
                    logger.info(
                        f"Auto-created fixed dataset 'my' from '{requested_dataset_name}': {dataset_abs_path}"
                    )
                except Exception as e:
                    return jsonify({
                        'error': f"Failed to prepare fixed dataset 'my' from '{requested_dataset_name}': {str(e)}"
                    }), 400
            else:
                return jsonify({
                    'error': f"Fixed dataset folder not found: {dataset_abs_path}. Please create/upload dataset first."
                }), 400

        # Preflight: textures path (relative to work_dir if needed)
        textures_arg = data.get('cc_textures_path', 'backgrounds')
        textures_abs_path = textures_arg if os.path.isabs(textures_arg) else os.path.join(work_dir, textures_arg)
        if not os.path.isdir(textures_abs_path):
            return jsonify({
                'error': f'Texture path not found: {textures_abs_path}'
            }), 400

        # Build environment - let venv activation handle PATH
        command_env = os.environ.copy()

        fixed_output_dir = DEFAULT_OUTPUT_DIR
        if os.path.exists(fixed_output_dir):
            logger.info(f'Removing stale output directory: {fixed_output_dir}')
            shutil.rmtree(fixed_output_dir)
        os.makedirs(fixed_output_dir, exist_ok=True)

        try:
            requested_max_samples = int(data.get('max_samples', 128))
        except (TypeError, ValueError):
            requested_max_samples = 128
        effective_max_samples = max(128, requested_max_samples)

        # Prepare command - use venv python module invocation when available.
        if use_venv_python_module:
            cmd = [
                BLENDERPROC_PYTHON,
                '-m', 'blenderproc',
                'run',
                blenderproc_script,
                bop_parent_path,
                fixed_dataset_name,
                textures_abs_path,
                fixed_output_dir,
                '--num_scenes', str(data.get('num_scenes', 10)),
                '--sample_size', str(data.get('sample_size', 100)),
                '--max_samples', str(effective_max_samples),
            ]
        else:
            cmd = [
                blenderproc_executable,
                'run',
                blenderproc_script,
                bop_parent_path,
                fixed_dataset_name,
                textures_abs_path,
                fixed_output_dir,
                '--num_scenes', str(data.get('num_scenes', 10)),
                '--sample_size', str(data.get('sample_size', 100)),
                '--max_samples', str(effective_max_samples),
            ]
        
        if data.get('seed'):
            cmd.extend(['--seed', str(data['seed'])])

        launch_in_terminal = bool(data.get('launch_in_terminal', False))

        if launch_in_terminal:
            terminal_process, exit_status_file, terminal_name, terminal_log_file = launch_generation_in_system_terminal(
                job_id,
                cmd,
                work_dir,
                command_env,
            )

            job_results[job_id] = {
                'status': 'running',
                'started_at': datetime.now().isoformat(),
                'parameters': {**data, 'output_dir': fixed_output_dir, 'max_samples': effective_max_samples},
                'effective_dataset_name': fixed_dataset_name,
                'output_dir': fixed_output_dir,
                'total_scenes': int(data.get('num_scenes', 10)),
                'completed_scenes': 0,
                'progress_percent': 0,
                'stdout': '',
                'stderr': '',
                'launch_mode': 'system-terminal',
                'terminal_name': terminal_name,
                'terminal_pid': terminal_process.pid,
                'exit_status_file': exit_status_file,
                'terminal_log_file': terminal_log_file,
            }

            logger.info(f"Started generation job {job_id} in system terminal '{terminal_name}'")

            return jsonify({
                'success': True,
                'job_id': job_id,
                'message': f"Training data generation launched in system terminal '{terminal_name}'",
                'launch_mode': 'system-terminal',
                'terminal_name': terminal_name,
                'output_dir': fixed_output_dir,
                'callback_enabled': False,
            }), 202

        job_results[job_id] = {
            'status': 'running',
            'started_at': datetime.now().isoformat(),
            'parameters': {**data, 'output_dir': fixed_output_dir, 'max_samples': effective_max_samples},
            'effective_dataset_name': fixed_dataset_name,
            'output_dir': fixed_output_dir,
            'total_scenes': int(data.get('num_scenes', 10)),
            'completed_scenes': 0,
            'progress_percent': 0,
            'stdout': '',
            'stderr': '',
            'launch_mode': 'backend-subprocess',
            'terminal_name': None,
            'terminal_pid': None,
            'exit_status_file': None,
            'terminal_log_file': None,
            'command': ' '.join(cmd),
        }

        worker = threading.Thread(
            target=run_blenderproc_job,
            args=(job_id, cmd, work_dir, command_env),
            daemon=True,
        )
        worker.start()

        logger.info(f"Started generation job {job_id} via backend subprocess channel")

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Training data generation started via backend subprocess channel',
            'launch_mode': 'backend-subprocess',
            'output_dir': fixed_output_dir,
            'callback_enabled': False,
        }), 202
        
    except Exception as e:
        logger.error(f"Error starting generation: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/callback', methods=['POST'])
@app.route('/generate/callback', methods=['POST'])
def receive_generation_callback():
    """Receive active status callback from BlenderProc main_ui_blend.py."""
    try:
        data = request.get_json(silent=True) or {}
        job_id = str(data.get('job_id') or '').strip()
        token = str(data.get('token') or '')
        status = str(data.get('status') or '').strip().lower()
        event = str(data.get('event') or '').strip().lower()
        message = str(data.get('message') or '')

        if not job_id:
            return jsonify({'error': 'Missing job_id'}), 400
        if job_id not in job_results:
            logger.warning(f"[Callback] Unknown job_id='{job_id}' from {request.remote_addr}")
            return jsonify({'error': 'Job not found'}), 404

        job_status = job_results[job_id]
        expected_token = str(job_status.get('callback_token') or '')
        if expected_token and not hmac.compare_digest(token, expected_token):
            logger.warning(f"[Callback] Invalid token for job_id='{job_id}' from {request.remote_addr}")
            return jsonify({'error': 'Invalid callback token'}), 403

        logger.info(
            f"[Callback] job_id={job_id} event={event} status={status} completed_scenes={data.get('completed_scenes')} total_scenes={data.get('total_scenes')}"
        )

        completed_scenes = data.get('completed_scenes')
        total_scenes = data.get('total_scenes', job_status.get('total_scenes') or 0)
        try:
            total_scenes = int(total_scenes)
        except Exception:
            total_scenes = int(job_status.get('total_scenes') or 0)

        previous_completed_scenes = int(job_status.get('completed_scenes') or 0)

        if completed_scenes is not None:
            try:
                completed_scenes = int(completed_scenes)
                completed_scenes = max(0, completed_scenes)
                completed_scenes = max(previous_completed_scenes, completed_scenes)
                job_status['completed_scenes'] = completed_scenes
            except Exception:
                completed_scenes = job_status.get('completed_scenes', 0)
        else:
            completed_scenes = job_status.get('completed_scenes', 0)

        if total_scenes > 0:
            progress_percent = int((int(completed_scenes) / total_scenes) * 100)
            progress_percent = max(0, min(100, progress_percent))
        else:
            progress_percent = job_status.get('progress_percent', 0)

        existing_status = str(job_status.get('status') or '').lower()
        terminal_states = {'completed', 'failed'}
        if status in ('running', 'completed', 'failed'):
            if existing_status in terminal_states and status == 'running':
                pass
            else:
                job_status['status'] = status

        job_status['last_callback_event'] = event
        job_status['last_callback_message'] = message
        job_status['total_scenes'] = total_scenes
        job_status['progress_percent'] = progress_percent
        job_status['callback_updated_at'] = datetime.now().isoformat()

        if status == 'completed':
            job_status['return_code'] = 0
            job_status['completed_at'] = job_status.get('completed_at') or datetime.now().isoformat()
            if total_scenes > 0 and int(completed_scenes) < total_scenes:
                job_status['completed_scenes'] = total_scenes
                job_status['progress_percent'] = 100
        elif status == 'failed':
            job_status['error'] = job_status.get('error') or message or 'Generation failed (callback)'
            job_status['completed_at'] = job_status.get('completed_at') or datetime.now().isoformat()

        job_results[job_id] = job_status

        return jsonify({'success': True, 'job_id': job_id, 'status': job_status.get('status')}), 200
    except Exception as e:
        logger.error(f"Error handling generation callback: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of a generation job using exit code as single source of truth."""
    try:
        if job_id not in job_results:
            return jsonify({
                'job_id': job_id,
                'status': 'not-found',
                'error': 'Job not found in backend memory (possibly restarted or stale job id).',
            }), 200
        
        job_status = job_results[job_id]

        # ===== EXIT CODE ONLY =====
        status = job_status.get('status')
        exit_code = parse_exit_code_from_file(job_status.get('exit_status_file'))
        if exit_code is not None:
            job_status['return_code'] = exit_code
            if exit_code == 0:
                status = 'completed'
                total_scenes = int(job_status.get('total_scenes') or 0)
                if total_scenes > 0:
                    job_status['completed_scenes'] = total_scenes
                    job_status['progress_percent'] = 100
            else:
                status = 'failed'
                job_status['error'] = job_status.get('error') or f'Generation exited with code {exit_code}'

        terminal_log_tail = read_terminal_log_tail(job_status)

        # Update job status in memory
        if status != job_status.get('status') or terminal_log_tail != job_status.get('stdout', ''):
            job_status.update({
                'status': status,
                'stdout': terminal_log_tail,
                'completed_at': job_status.get('completed_at') or (datetime.now().isoformat() if status in ('completed', 'failed') else None)
            })
            job_results[job_id] = job_status

        log_tail_chars = 30000

        # ===== SIMPLIFIED RESPONSE: Only status, no scene progress =====
        return jsonify({
            'job_id': job_id,
            'status': status,  # 'running' | 'completed' | 'failed'
            'started_at': job_status.get('started_at'),
            'completed_at': job_status.get('completed_at'),
            'return_code': job_status.get('return_code'),
            'completed_scenes': job_status.get('completed_scenes'),
            'total_scenes': job_status.get('total_scenes'),
            'progress_percent': job_status.get('progress_percent'),
            'last_callback_event': job_status.get('last_callback_event'),
            'last_callback_message': job_status.get('last_callback_message'),
            'callback_updated_at': job_status.get('callback_updated_at'),
            'launch_mode': job_status.get('launch_mode'),
            'terminal_name': job_status.get('terminal_name'),
            'terminal_pid': job_status.get('terminal_pid'),
            'stdout': (terminal_log_tail or job_status.get('stdout', ''))[-log_tail_chars:],
            'stderr': job_status.get('stderr', '')[-log_tail_chars:],
            'error': job_status.get('error'),
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/paths', methods=['GET'])
def get_default_paths():
    """Get suggested default paths for the system."""
    return jsonify({
        'workspace_root': str(AUTOMATED_DIR),
        'bop_parent_path': BOP_PARENT_PATH,
        'cc_textures_path': os.path.join(BOP_PARENT_PATH, 'resources', 'cctextures'),
        'output_dir': DEFAULT_OUTPUT_DIR,
        'default_dataset': 'hb',
        'example_datasets': ['hb', 'tless', 'ycbv', 'tyol']
    }), 200


# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum file size is 500 MB'}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting 6D Detection UI Backend Server...")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)