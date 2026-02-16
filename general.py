import contextlib
import copy
import datetime
import hashlib
import json
import os
import random
import string
import subprocess
import time
from dataclasses import dataclass, fields, replace

import matplotlib.pyplot as plt
import numpy as np

with contextlib.suppress(BaseException):
    import torch # type: ignore # noqa: I001


@dataclass
class GeneralConfig:
    verbose_level: int = 5
    plot_level: int = 5
    figs_dir: str = "./figs/"
    logs_dir: str = "./logs/"
    data_dir: str = "./data/"
    random_str: str = ""

    def update_from_config(self, config):
        """
        Update all existing parameters of this config from another config
        object or a dict. Extra keys/attributes are ignored.
        """
        if isinstance(config, dict):
            source = config
            for f in fields(self):
                if f.name in source:
                    setattr(self, f.name, source[f.name])
        else:
            # assume config is an object with attributes
            for f in fields(self):
                if hasattr(config, f.name):
                    setattr(self, f.name, getattr(config, f.name))
        return self

    def update_from_json(self, file_path):
        """
        Update all existing parameters of this config from a JSON file. Extra keys are ignored.
        """
        with open(file_path) as json_file:
            json_dict = json.load(json_file)
        return self.update_from_config(json_dict)

    def copy(self):
        return copy.deepcopy(self)


class General:
    """
    A class used to represent general utility functions and configurations.
    Attributes
    """

    def __init__(self, config: GeneralConfig, **overrides):
        # strict override: only allow existing fields
        allowed = set(config.__dataclass_fields__.keys())
        unknown = set(overrides) - allowed
        if unknown:
            raise TypeError(f"Unknown parameter(s): {sorted(unknown)}")

        self.config = replace(config, **overrides)  # makes a new config

    def create_dirs(self, dir_list=()):
        """
        Creates directories for figures, logs, and data if they do not exist.
        This method checks if the directories specified by `self.config.figs_dir`, `self.config.logs_dir`,
        and `self.config.data_dir` exist. If they do not exist, it creates them.
        Returns:
            None
        """
        os.makedirs(self.config.figs_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
        for d in dir_list:
            os.makedirs(d, exist_ok=True)

    def copy(self):
        return copy.deepcopy(self)

    def print(self, text="", thr=0):
        """
        Prints the given text if the verbose level is greater than or equal to the specified threshold.
        Args:
            text (str): The text to be printed. Defaults to an empty string.
            thr (int): The threshold level for verbosity. The text will be printed only if
            the instance's verbose_level is greater than or equal to this value. Defaults to 0.
        """
        if self.config.verbose_level >= thr:
            print(text)

    def gen_random_str(self, length=6):
        """
        Generates a random string of specified length.
        This method creates a random string consisting of ASCII letters and digits.
        The generated string is stored in the instance variable `random_str` and
        also printed out.
        Args:
            length (int, optional): The length of the random string to generate.
                                    Defaults to 6.
        Returns:
            str: The generated random string.
        """

        letters = string.ascii_letters + string.digits
        self.config.random_str = "".join(random.choice(letters) for i in range(length))
        self.print(f"Random string for this run: {self.config.random_str}", thr=0)
        return self.config.random_str

    def print_info(self, params):
        """
        Prints the provided parameters along with the current date, time, and the latest Git commit ID.
        Args:
            params (dict): A dictionary of parameters to be printed.
        The method performs the following steps:
        1. Prints the provided parameters using the `print_params` method.
        2. Retrieves the current date and time.
        3. Attempts to get the latest Git commit ID. If the Git command fails or no Git repository is found, it sets the commit ID to an error message.
        4. Prints the current date and time.
        5. Prints the latest Git commit ID.
        """

        self.print_params(params)

        now = datetime.datetime.now()
        current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        # Get the latest Git commit ID
        try:
            latest_commit_id = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            )
        except subprocess.CalledProcessError:
            latest_commit_id = "No Git repository found or Git command failed."

        # Print the formatted date, time, and latest commit ID
        self.print(f"Current Date and Time: {current_datetime}", thr=0)
        self.print(f"Latest Git Commit ID: {latest_commit_id}", thr=0)

    def print_params(self, params=None):
        """
        Prints the attributes of the given params object.
        Args:
            params (object): An object whose attributes will be printed.
            Only non-callable attributes that do not start with '__' will be printed.
        """
        if params is None:
            params = self

        self.print("Run parameters:", thr=0)
        for attr in dir(params):
            if not callable(getattr(params, attr)) and not attr.startswith("__"):
                self.print(f"{attr} = {getattr(params, attr)}", thr=0)
        self.print("\n", thr=0)

    def set_seed(self, seed=None, to_set=("numpy",)):
        """
        Sets the random seed for reproducibility.
        Args:
            seed (int): The seed value to set. If None, a random seed will be generated.
            to_set (tuple): A tuple of libraries to set the seed for. Default is ("numpy",).
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self.print(f"Random seed: {seed}", thr=5)
        for lib in to_set:
            if lib == "numpy":
                np.random.seed(seed)
            elif lib == "torch":
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            elif lib == "cupy":
                import cupy as cp   # type: ignore # noqa: I001

                cp.random.seed(seed)
            elif lib == "tensorflow":
                import tensorflow as tf # type: ignore # noqa: I001

                tf.random.set_seed(seed)
            elif lib == "sionna":
                import sionna   # type: ignore # noqa: I001

                sionna.phy.config.seed = seed
            else:
                self.print(f"Library '{lib}' not recognized. Seed not set.", thr=1)

    def import_attributes(self, params, obj=None, add_atts=True, overwrite=True):
        """
        Imports attributes from the given params object to the specified object.
        Args:
            obj (object): The object to which attributes will be imported.
            params (object): The object from which attributes will be imported.
        """
        if obj is None:
            obj = self
        for attr in dir(params):
            if not callable(getattr(params, attr)) and not attr.startswith("__"):
                if hasattr(obj, attr) or add_atts:
                    if overwrite:
                        setattr(obj, attr, getattr(params, attr))
                        self.print(f"Attribute '{attr}' imported from params to {obj}.", thr=6)
                    else:
                        self.print(
                            f"Attribute '{attr}' already exists in {obj}. Skipping import.", thr=6
                        )
                else:
                    self.print(f"Attribute '{attr}' not found in {obj}. Importing.", thr=6)

    def unique_list(self, input_list):
        """
        Remove duplicates from the input list while preserving the order.
        Args:
            input_list (list): The list from which duplicates need to be removed.
        Returns:
            list: A new list with duplicates removed, preserving the original order.
        """

        seen = set()
        input_list = [x for x in input_list if not (x in seen or seen.add(x))]
        return input_list

    def save_dict_to_json(self, dictionary, file_path):
        """Save a dictionary to a JSON file."""
        for key, value in dictionary.items():
            # Check if the value is JSON serializable
            try:
                json.dumps(value)
            except (TypeError, OverflowError):
                self.print(
                    f"Value {value} for key '{key}' is not JSON serializable and will be skipped.",
                    thr=1,
                )
                continue
        with open(file_path, "w") as json_file:
            json.dump(dictionary, json_file, indent=4)

    def load_dict_from_json(self, file_path, convert_values=False):
        """Load a dictionary from a JSON file."""

        def convert_str_to_number(value):
            """Convert string to int or float if possible."""
            try:
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value

        def convert_dict_str_to_number(dictionary):
            """Convert all string keys in a dictionary to numbers."""
            if not isinstance(dictionary, dict):
                return dictionary
            return {
                convert_str_to_number(key): convert_dict_str_to_number(value)
                for key, value in dictionary.items()
            }

        with open(file_path) as json_file:
            json_dict = json.load(json_file)
        json_dict_updated = convert_dict_str_to_number(json_dict) if convert_values else json_dict

        return json_dict_updated

    def save_class_attributes_to_json(self, obj, file_path):
        """
        Save all attributes of a class instance to a JSON file.
        Args:
            obj (object): The class instance whose attributes need to be saved.
            file_path (str): The path to the JSON file where attributes will be saved.
        """
        attributes = {
            attr: getattr(obj, attr)
            for attr in dir(obj)
            if not callable(getattr(obj, attr)) and not attr.startswith("__")
        }
        # Check if the attribute is JSON serializable
        serializable_attributes = {}
        for attr, value in attributes.items():
            try:
                json.dumps(value)
                serializable_attributes[attr] = value
            except (TypeError, OverflowError):
                self.print(
                    f"Attribute '{attr}' is not JSON serializable and will be skipped.", thr=1
                )
        self.save_dict_to_json(serializable_attributes, file_path)

    def load_class_attributes_from_json(self, obj, file_path):
        """
        Load all attributes of a class instance from a JSON file.
        Args:
            file_path (str): The path to the JSON file from which attributes will be loaded.
        """
        attributes = self.load_dict_from_json(file_path)
        for attr, value in attributes.items():
            if hasattr(obj, attr):
                setattr(obj, attr, value)
            else:
                self.print(f"Attribute '{attr}' not found in the object {obj}", thr=1)

    def compute_hash(self, file_path):
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


@dataclass
class GeneralParallelConfig(GeneralConfig):
    import_cupy: bool = False
    use_cupy: bool = False
    gpu_id: int = 0
    use_torch: bool = False


class GeneralParallel(General):
    def __init__(self, config: GeneralParallelConfig, **overrides):
        # strict override: only allow existing fields
        super().__init__(config, **overrides)

        if self.config.use_torch:
            self.init_device_torch()
        else:
            self.device = None

    def init_device_torch(self):
        """
        Initializes the PyTorch device for the current instance.
        This method sets the `device` attribute to 'cuda' if a GPU is available,
        otherwise it sets it to 'cpu'. It also prints the selected device.
        Returns:
            None
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.print(f"Torch device: {self.device}", thr=0)

    def cupy_plt_plot(self, *args, **kwargs):
        """
        Plots data using Matplotlib after transferring CuPy arrays to NumPy arrays.
        This method takes any number of positional and keyword arguments, transfers
        the first two positional arguments from CuPy arrays to NumPy arrays, and then
        plots them using Matplotlib's `plot` function.
        Parameters:
        *args: list
            Positional arguments to be passed to Matplotlib's `plot` function. The first
            two arguments, if present, are expected to be CuPy arrays and will be transferred
            to NumPy arrays.
        **kwargs: dict
            Keyword arguments to be passed to Matplotlib's `plot` function.
        Returns:
        None
        """

        args = list(args)

        # Apply np.sqrt to the first two arguments
        if len(args) > 0:
            args[0] = self.numpy_transfer(args[0], dst="numpy")
        if len(args) > 1:
            args[1] = self.numpy_transfer(args[1], dst="numpy")

        plt.plot(*args, **kwargs)
        # plt.show()

    def cupy_gpu_init(self):
        """
        Initializes GPU settings using CuPy if enabled.
        This method performs the following steps:
        1. Checks GPU usage.
        2. Prints GPU memory status.
        3. Warms up the GPU.
        4. Compares GPU and CPU performance.
        Preconditions:
        - `self.config.use_cupy` must be True to enable CuPy usage.
        - `self.config.import_cupy` must be True to allow CuPy import.
        Raises:
        - RuntimeError: If GPU initialization fails at any step.
        """

        if self.config.use_cupy and self.config.import_cupy:
            self.check_gpu_usage()
            self.print_gpu_memory()
            self.warm_up_gpu()
            self.gpu_cpu_compare()

    def check_cupy_gpu(self, gpu_id=0):
        """
        Checks if CuPy is installed and if a specified GPU is available.
        This method attempts to import CuPy and check for the availability of a GPU with the given ID.
        It prints the CuPy version, the number of GPUs available, and the properties of the specified GPU.
        Args:
            gpu_id (int): The ID of the GPU to check. Default is 0.
        Returns:
            bool: True if CuPy is installed and the specified GPU is available, False otherwise.
        """

        if not self.config.import_cupy:
            return False
        try:
            import cupy as cp   # type: ignore # noqa: I001

            # Check if CuPy is installed
            self.print(f"CuPy version: {cp.__version__}", thr=0)

            num_gpus = cp.cuda.runtime.getDeviceCount()
            self.print(f"Number of GPUs available: {num_gpus}", thr=0)

            # Check if the GPU is available
            # cp.cuda.Device(gpu_id).compute_capability
            self.print(f"GPU {gpu_id} is available", thr=0)

            self.print(
                f"GPU {gpu_id} properties: {cp.cuda.runtime.getDeviceProperties(gpu_id)}",
                thr=0,
            )
            return True
        except ImportError:
            self.print("CuPy is not installed.", thr=0)
        except Exception:
            self.print("GPU is not available or CUDA is not installed correctly.", thr=0)
        return False

    def get_gpu_device(self):
        """
        Get the GPU device.
        This method returns the GPU device if the use of CuPy is enabled.
        If CuPy is not enabled, it returns None.
        Returns:
            cp.cuda.Device or None: The GPU device if CuPy is enabled, otherwise None.
        """

        if self.config.use_cupy:
            import cupy as cp   # type: ignore # noqa: I001

            return cp.cuda.Device(self.config.gpu_id)
        else:
            return None

    def check_gpu_usage(self):
        """
        Check and print the current GPU usage.
        This method uses the CuPy library to access the GPU device specified by
        `self.config.gpu_id` and prints the current device information.
        Returns:
            None
        """

        import cupy as cp   # type: ignore # noqa: I001

        with cp.cuda.Device(self.config.gpu_id) as device:
            self.print(f"Current device: {device}", 0)

    def print_gpu_memory(self):
        """
        Prints the used and total GPU memory for the specified GPU device.
        This method uses CuPy to access the GPU memory pool and prints the
        used and total memory in bytes for the GPU device specified by
        `self.config.gpu_id`.
        Requires:
            - CuPy library installed.
            - `self.config.gpu_id` to be set to a valid GPU device ID.
        Prints:
            - Used GPU memory in bytes.
            - Total GPU memory in bytes.
        """

        import cupy as cp   # type: ignore # noqa: I001

        with cp.cuda.Device(self.config.gpu_id):
            mempool = cp.get_default_memory_pool()
            self.print(f"Used GPU memory: {mempool.used_bytes()} bytes", thr=0)
            self.print(f"Total GPU memory: {mempool.total_bytes()} bytes", thr=0)

    # Initialize and warm-up
    def warm_up_gpu(self):
        """
        Warm up the GPU by performing a series of operations.
        This method initializes the GPU by performing several operations using the CuPy library.
        It creates arrays, performs matrix multiplications, and synchronizes the GPU stream to
        ensure that the GPU is ready for subsequent computations. The time taken for the warmup
        process is printed.
        Returns:
            None
        """

        self.print("Starting GPU warmup.", thr=0)
        import cupy as cp   # type: ignore # noqa: I001

        with cp.cuda.Device(self.config.gpu_id):
            start = time.time()
            _ = cp.array([1, 2, 3])
            _ = cp.array([4, 5, 6])
            a = cp.random.rand(1000, 1000)
            _ = cp.dot(cp.array([[1, 2], [3, 4]]), cp.array([[5, 6], [7, 8]]))
            _ = cp.dot(a, a)
            cp.cuda.Stream.null.synchronize()
            end = time.time()
        self.print(f"GPU warmup time: {end - start}", thr=0)

    # Perform computation
    def gpu_cpu_compare(self, size=20000):
        """
        Compares the computation time of matrix multiplication on CPU and GPU.
        This method generates two random matrices of the specified size, performs
        matrix multiplication on both CPU and GPU, and measures the time taken for
        each operation. The results are printed and returned.
        Parameters:
        size (int): The size of the generated square matrices. Default is 20000.
        Returns:
        tuple: A tuple containing the GPU computation time and CPU computation time.
        """

        self.print("Starting CPU and GPU times compare.", thr=0)
        import cupy as cp   # type: ignore # noqa: I001
        import numpy

        # Generate data
        a_cpu = numpy.random.rand(size, size).astype(float)
        b_cpu = numpy.random.rand(size, size).astype(float)

        # Measure CPU time for comparison
        start = time.time()
        _ = numpy.dot(a_cpu, b_cpu)
        end = time.time()
        cpu_time = end - start
        self.print(f"CPU time: {cpu_time}", thr=0)

        with cp.cuda.Device(self.config.gpu_id):
            # Transfer data to GPU
            a_gpu = cp.asarray(a_cpu)
            b_gpu = cp.asarray(b_cpu)

            # Measure GPU time
            start = time.time()
            _ = cp.dot(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize()  # Ensure all computations are finished
            end = time.time()
            gpu_time = end - start
            self.print(f"GPU time: {gpu_time}", thr=0)

        return gpu_time, cpu_time

    def numpy_transfer(self, arrays, dst="numpy"):
        """
        Transfers arrays between NumPy and CuPy contexts.
        Parameters:
        -----------
        arrays : list or array-like
            The input arrays to be transferred. Can be a list of arrays or a single array.
        dst : str, optional
            The destination context for the arrays. Can be 'numpy' to transfer to NumPy arrays
            or 'context' to transfer to CuPy arrays. Default is 'numpy'.
        Returns:
        --------
        out : list or array-like
            The transferred arrays in the specified destination context. If the input was a list,
            the output will be a list of arrays. If the input was a single array, the output will
            be a single array.
        Notes:
        ------
        - If `self.config.import_cupy` is False, the input arrays are returned as-is without any transfer.
        - If `dst` is 'numpy' and the input arrays are not NumPy arrays, they are converted to NumPy arrays.
        - If `dst` is 'context' and the input arrays are not CuPy arrays, they are converted to CuPy arrays.
        """
        import numpy

        if self.config.import_cupy:
            if isinstance(arrays, list):
                out = []
                for i in range(len(arrays)):
                    if dst == "numpy" and not isinstance(arrays[i], numpy.ndarray):
                        out.append(np.asnumpy(arrays[i]))
                    elif dst == "context" and not isinstance(arrays[i], np.ndarray):
                        out.append(np.asarray(arrays[i]))
            else:
                if dst == "numpy" and not isinstance(arrays, numpy.ndarray):
                    out = np.asnumpy(arrays)
                elif dst == "context" and not isinstance(arrays, np.ndarray):
                    out = np.asarray(arrays)
        else:
            out = arrays
        return out
