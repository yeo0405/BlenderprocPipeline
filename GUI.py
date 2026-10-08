import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import json
import shutil

try:
    from tomo_system.env_manager import get_env_manager
except Exception:
    get_env_manager = None

ENV_NAME = "Data_Generation"
REQUIREMENTS_PATH = "requirements.txt"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BLENDER_CONFIG = {
    "python": os.path.join(BASE_DIR, ".venv_blender5", "bin", "python"),
    "blender": "/home/yeo/Downloads/blender-5.2.1-linux-x64",
}

DEFAULTS = {
    "output_dir": "",
    "num_scenes": "200",
    "sample_size": "100",
}


class DataGenerationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Generation")
        self.root.geometry("1000x760")
        self.log_queue = queue.Queue()
        self.worker_thread = None
        self.worker_process = None
        self.blend_files = []
        self.base_dir = BASE_DIR
        self.target_python = sys.executable

        print(f"[INFO] GUI python: {self.target_python}")

        self._build_ui()
        self.root.after(100, self._poll_logs)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="both", expand=True)

        ttk.Label(
            top,
            text="Data Generation",
            font=("Arial", 18, "bold")
        ).pack(anchor="w", pady=(0, 10))

        path_box = ttk.LabelFrame(
            top, text="Recommended Inputs", padding=12
        )
        path_box.columnconfigure(1, weight=1)
        path_box.pack(fill="x", pady=8)

        self.output_dir_var = tk.StringVar(
            value=DEFAULTS["output_dir"]
        )
        self._add_path_row(
            path_box, 0, "Output Dir", self.output_dir_var
        )

        blend_row = ttk.Frame(path_box)
        blend_row.grid(
            row=1, column=0, columnspan=3,
            sticky="ew", pady=5
        )

        ttk.Label(
            blend_row,
            text=".blend Files"
        ).pack(side="left", padx=(0, 8))

        ttk.Button(
            blend_row,
            text="Select .blend Files",
            command=self._select_blend_files
        ).pack(side="left", padx=4)

        ttk.Button(
            blend_row,
            text="Clear Selection",
            command=self._clear_blend_files
        ).pack(side="left", padx=4)

        self.blend_info_var = tk.StringVar(
            value="No .blend files selected"
        )
        ttk.Label(
            path_box,
            textvariable=self.blend_info_var,
            foreground="gray"
        ).grid(
            row=2, column=0, columnspan=3,
            sticky="w", pady=(5, 0)
        )

        self.blend_list = tk.Listbox(
            path_box,
            height=6,
            width=120
        )
        self.blend_list.grid(
            row=3, column=0, columnspan=3,
            sticky="ew", pady=(8, 0)
        )

        gen_box = ttk.LabelFrame(
            top, text="Generation Settings", padding=12
        )
        gen_box.pack(fill="x", pady=8)

        self.num_scenes_var = tk.StringVar(
            value=DEFAULTS["num_scenes"]
        )
        self.sample_size_var = tk.StringVar(
            value=DEFAULTS["sample_size"]
        )
        self.cpu_mode_var = tk.StringVar(value="normal")

        self._add_text_row(
            gen_box, 0, "Num Scenes", self.num_scenes_var
        )
        self._add_text_row(
            gen_box, 1, "Sample Size", self.sample_size_var
        )

        ttk.Label(
            gen_box,
            text="Blender Version"
        ).grid(
            row=2, column=0,
            sticky="w",
            padx=(0, 8),
            pady=5
        )

        ttk.Label(
            gen_box,
            text="Blender 5.2.1"
        ).grid(
            row=2, column=1,
            sticky="w",
            pady=5
        )

        cpu_box = ttk.LabelFrame(
            gen_box, text="CPU Usage", padding=8
        )
        cpu_box.grid(
            row=3, column=0, columnspan=2,
            sticky="ew", pady=(8, 5)
        )

        ttk.Radiobutton(
            cpu_box,
            text="Normal",
            value="normal",
            variable=self.cpu_mode_var
        ).pack(side="left", padx=8)

        ttk.Radiobutton(
            cpu_box,
            text="Low (Slower)",
            value="low",
            variable=self.cpu_mode_var
        ).pack(side="left", padx=8)

        ttk.Radiobutton(
            cpu_box,
            text="Very Low (Much Slower)",
            value="very_low",
            variable=self.cpu_mode_var
        ).pack(side="left", padx=8)

        btn_box = ttk.Frame(top)
        btn_box.pack(fill="x", pady=8)

        ttk.Button(
            btn_box,
            text="Start",
            command=self._start_generation
        ).pack(side="left", padx=4)

        ttk.Button(
            btn_box,
            text="Stop",
            command=self._stop_generation
        ).pack(side="left", padx=4)

        ttk.Button(
            btn_box,
            text="Clear Log",
            command=self._clear_log
        ).pack(side="left", padx=4)

        log_box = ttk.LabelFrame(
            top, text="Console", padding=8
        )
        log_box.pack(fill="both", expand=True, pady=8)

        self.console = scrolledtext.ScrolledText(
            log_box,
            height=18,
            font=("Courier", 10)
        )
        self.console.pack(fill="both", expand=True)
        self.console.insert("end", "GUI started.\n")
        self.console.configure(state="disabled")

    def _add_text_row(self, parent, row, label, variable):
        ttk.Label(
            parent,
            text=label
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=5
        )

        ttk.Entry(
            parent,
            textvariable=variable,
            width=70
        ).grid(
            row=row,
            column=1,
            sticky="w",
            pady=5
        )

    def _add_path_row(self, parent, row, label, variable):
        ttk.Label(
            parent,
            text=label
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=5
        )

        ttk.Entry(
            parent,
            textvariable=variable
        ).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=5
        )

        ttk.Button(
            parent,
            text="Browse",
            command=lambda: self._browse_folder(variable)
        ).grid(
            row=row,
            column=2,
            padx=6,
            pady=5
        )

    def _browse_folder(self, variable):
        folder = filedialog.askdirectory()
        if folder:
            variable.set(folder)

    def _select_blend_files(self):
        files = filedialog.askopenfilenames(
            title="Select .blend Files",
            filetypes=[
                ("Blender files", "*.blend"),
                ("All files", "*.*")
            ]
        )

        if not files:
            return

        self.blend_files = list(files)
        self._refresh_blend_list()
        self._log(
            f"Selected {len(self.blend_files)} .blend file(s)"
        )

    def _clear_blend_files(self):
        self.blend_files = []
        self._refresh_blend_list()
        self._log("Cleared .blend file selection")

    def _refresh_blend_list(self):
        self.blend_list.delete(0, tk.END)

        if not self.blend_files:
            self.blend_info_var.set(
                "No .blend files selected"
            )
            return

        self.blend_info_var.set(
            f"{len(self.blend_files)} .blend file(s) selected"
        )

        for file_path in self.blend_files:
            self.blend_list.insert(
                tk.END,
                file_path
            )

    def _validate_inputs(self):
        errors = []

        if not self.output_dir_var.get().strip():
            errors.append("Output Dir is required")

        if not self.blend_files:
            errors.append(
                "Please select at least one .blend file"
            )

        for field_name, value in [
            ("num_scenes", self.num_scenes_var.get()),
            ("sample_size", self.sample_size_var.get()),
        ]:
            try:
                value_int = int(value)

                if value_int <= 0:
                    errors.append(
                        f"{field_name} must be greater than 0"
                    )
            except Exception:
                errors.append(
                    f"{field_name} must be an integer"
                )

        for file_path in self.blend_files:
            if not file_path.lower().endswith(".blend"):
                errors.append(
                    f"Invalid file selected: {file_path}"
                )

        if not os.path.isfile(
            os.path.join(BLENDER_CONFIG["blender"], "blender")
        ):
            errors.append(
                f"Blender executable not found: "
                f"{os.path.join(BLENDER_CONFIG['blender'], 'blender')}"
            )

        if not os.path.isfile(BLENDER_CONFIG["python"]):
            errors.append(
                f"Python environment not found: "
                f"{BLENDER_CONFIG['python']}"
            )

        if errors:
            messagebox.showerror(
                "Validation Error",
                "\n".join(errors)
            )
            return False

        dataset_path = os.path.join(
            self.base_dir,
            "bop",
            "BlenderProc",
            "hb"
        )

        hb_models_dir = os.path.join(
            dataset_path,
            "models"
        )

        temp_dir = os.path.join(
            self.base_dir,
            "bop",
            "BlenderProc",
            "temp"
        )

        try:
            os.makedirs(
                hb_models_dir,
                exist_ok=True
            )

            for fname in os.listdir(hb_models_dir):
                if fname.lower().endswith(".blend"):
                    fpath = os.path.join(
                        hb_models_dir,
                        fname
                    )

                    try:
                        os.remove(fpath)
                        self._log(
                            f"Removed old blend: {fname}"
                        )
                    except Exception as e:
                        self._log(
                            f"Failed to remove old blend "
                            f"{fname}: {e}"
                        )

            for blend_file in self.blend_files:
                filename = os.path.basename(blend_file)

                dest_path = os.path.join(
                    hb_models_dir,
                    filename
                )

                shutil.copy2(
                    blend_file,
                    dest_path
                )

                self._log(
                    f"Copied: {filename} -> {hb_models_dir}"
                )

            self._log(
                f"All .blend files copied to "
                f"{hb_models_dir}"
            )

            temp_camera_path = os.path.join(
                temp_dir,
                "camera.json"
            )

            camera_path = os.path.join(
                dataset_path,
                "camera_primesense.json"
            )

            if os.path.exists(temp_camera_path):
                shutil.copy2(
                    temp_camera_path,
                    camera_path
                )
                self._log(
                    "Copied camera.json from temp"
                )
            else:
                default_camera = {
                    "fx": 572.4114,
                    "fy": 572.4114,
                    "cx": 325.2611,
                    "cy": 242.0449,
                    "width": 640,
                    "height": 480,
                    "depth_scale": 0.001
                }

                with open(
                    camera_path,
                    "w"
                ) as f:
                    json.dump(
                        default_camera,
                        f,
                        indent=2
                    )

                self._log(
                    "Created default camera_primesense.json"
                )

            temp_targets_path = os.path.join(
                temp_dir,
                "test_targets_bop19.json"
            )

            targets_path = os.path.join(
                dataset_path,
                "test_targets_bop19.json"
            )

            if os.path.exists(temp_targets_path):
                shutil.copy2(
                    temp_targets_path,
                    targets_path
                )

                self._log(
                    "Copied test_targets_bop19.json from temp"
                )
            else:
                default_targets = [
                    {
                        "obj_id": 1,
                        "inst_count": 1
                    }
                ]

                with open(
                    targets_path,
                    "w"
                ) as f:
                    json.dump(
                        default_targets,
                        f,
                        indent=2
                    )

                self._log(
                    "Created default test_targets_bop19.json"
                )

        except Exception as exc:
            messagebox.showerror(
                "File Setup Error",
                f"Failed to setup dataset: {exc}"
            )

            self._log(
                f"Dataset setup error: {exc}"
            )

            return False

        self._log(
            "Input validation passed"
        )

        return True

    def _start_generation(self):
        if not self._validate_inputs():
            return

        if (
            self.worker_thread
            and self.worker_thread.is_alive()
        ):
            self._log(
                "A task is already running"
            )
            return

        self._log("=" * 60)
        self._log("Start clicked")
        self._log("=" * 60)

        self.worker_thread = threading.Thread(
            target=self._worker_main,
            daemon=True
        )

        self.worker_thread.start()

    def _worker_main(self):
        try:
            script_path = os.path.join(
                self.base_dir,
                "bop",
                "BlenderProc",
                "examples",
                "datasets",
                "bop_challenge",
                "main_ui.py"
            )

            if not os.path.exists(script_path):
                self._log(
                    f"Script not found: {script_path}"
                )
                return

            bop_parent_path = os.path.join(
                self.base_dir,
                "bop",
                "BlenderProc"
            )

            textures_path = os.path.join(
                bop_parent_path,
                "backgrounds"
            )

            dataset_name = "hb"

            output_dir_abs = os.path.abspath(
                self.output_dir_var.get().strip()
            )

            if not os.path.isdir(
                bop_parent_path
            ):
                self._log(
                    f"BOP parent path not found: "
                    f"{bop_parent_path}"
                )
                return

            dataset_dir = os.path.join(
                bop_parent_path,
                dataset_name
            )

            if not os.path.isdir(dataset_dir):
                self._log(
                    f"Dataset folder not found: "
                    f"{dataset_dir}"
                )
                return

            if not os.path.isdir(textures_path):
                self._log(
                    f"Textures path not found: "
                    f"{textures_path}"
                )
                return

            os.makedirs(
                output_dir_abs,
                exist_ok=True
            )

            venv_bin = os.path.dirname(
                BLENDER_CONFIG["python"]
            )

            blenderproc_exe = os.path.join(
                venv_bin,
                "blenderproc"
            )

            if not os.path.isfile(
                blenderproc_exe
            ):
                self._log(
                    "Cannot find blenderproc executable:"
                )
                self._log(
                    blenderproc_exe
                )
                return

            blender_exe = os.path.join(
                BLENDER_CONFIG["blender"],
                "blender"
            )

            cpu_mode = self.cpu_mode_var.get()

            if cpu_mode == "low":
                cpu_threads = 4
                physics_substeps = 8
                physics_solver_iters = 12
            elif cpu_mode == "very_low":
                cpu_threads = 2
                physics_substeps = 4
                physics_solver_iters = 8
            else:
                cpu_threads = None
                physics_substeps = 20
                physics_solver_iters = 25

            env = os.environ.copy()

            if cpu_threads is not None:
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    env[name] = str(cpu_threads)

                env["BLENDER_THREADS"] = str(
                    cpu_threads
                )
                env["PHYSICS_SUBSTEPS"] = str(
                    physics_substeps
                )
                env["PHYSICS_SOLVER_ITERS"] = str(
                    physics_solver_iters
                )
            else:
                env.pop(
                    "BLENDER_THREADS",
                    None
                )
                env.pop(
                    "PHYSICS_SUBSTEPS",
                    None
                )
                env.pop(
                    "PHYSICS_SOLVER_ITERS",
                    None
                )

            self._log(
                "[Blender] Blender 5.2.1"
            )

            self._log(
                f"[Blender path] {blender_exe}"
            )

            self._log(
                f"[BlenderProc] {blenderproc_exe}"
            )

            if cpu_threads is not None:
                self._log(
                    f"[CPU] Mode: {cpu_mode}, "
                    f"threads={cpu_threads}, "
                    f"physics={physics_substeps}/"
                    f"{physics_solver_iters}"
                )
            else:
                self._log(
                    "[CPU] Mode: normal"
                )

            cmd = [
                blenderproc_exe,
                "run",
                "--custom-blender-path",
                BLENDER_CONFIG["blender"],
                script_path,
                bop_parent_path,
                dataset_name,
                textures_path,
                output_dir_abs,
                "--num_scenes",
                self.num_scenes_var.get().strip(),
                "--sample_size",
                self.sample_size_var.get().strip(),
                "--max_samples",
                "128",
            ]

            self._log("Running:")
            self._log(" ".join(cmd))

            self.worker_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(script_path),
                env=env,
                bufsize=1
            )

            if self.worker_process.stdout:
                for line in self.worker_process.stdout:
                    line = line.rstrip()

                    if line:
                        self._log(line)

            self.worker_process.wait()

            self._log(
                f"Process exited with "
                f"{self.worker_process.returncode}"
            )

        except Exception as exc:
            self._log(
                f"Worker error: {exc}"
            )

        finally:
            self.worker_process = None

    def _stop_generation(self):
        if self.worker_process:
            try:
                self.worker_process.terminate()
                self._log(
                    "Process terminate requested"
                )
            except Exception as exc:
                self._log(
                    f"Stop error: {exc}"
                )
        else:
            self._log(
                "No active process to stop"
            )

    def _clear_log(self):
        self.console.configure(
            state="normal"
        )
        self.console.delete(
            "1.0",
            "end"
        )
        self.console.configure(
            state="disabled"
        )

    def _log(self, text):
        self.log_queue.put(text)

    def _poll_logs(self):
        try:
            while True:
                line = self.log_queue.get_nowait()

                self.console.configure(
                    state="normal"
                )
                self.console.insert(
                    "end",
                    line + "\n"
                )
                self.console.see("end")
                self.console.configure(
                    state="disabled"
                )
        except queue.Empty:
            pass

        self.root.after(
            100,
            self._poll_logs
        )


def main():
    root = tk.Tk()
    app = DataGenerationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()