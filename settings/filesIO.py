import os
import numpy as np


class FieldIO:
    def __init__(self, base_path: str):
        self.base_path = base_path

    @staticmethod
    def _check_path(path: str):
        if not os.path.isdir(path):
            os.makedirs(path)

    def _drift_path(self):
        path = os.path.join(self.base_path, "drift")
        self._check_path(path)
        return path

    def _weighting_path(self, view: str):
        path = os.path.join(self.base_path, "weighting", view)
        self._check_path(path)
        return path

    def save_field(
        self,
        field,
        Ex,
        Ey,
        Ez,
        namefile: str,
        type_field: str,
        view: str | None = None,
    ):
        if type_field == "drift":
            path = self._drift_path()

        elif type_field == "weighting":
            if view is None:
                raise ValueError("view must be provided for weighting field")
            path = self._weighting_path(view)

        else:
            raise ValueError("type_field must be 'drift' or 'weighting'")

        np.save(os.path.join(path, f"{namefile}.npy"), field)
        np.save(os.path.join(path, f"{namefile}_Ex.npy"), Ex)
        np.save(os.path.join(path, f"{namefile}_Ey.npy"), Ey)
        np.save(os.path.join(path, f"{namefile}_Ez.npy"), Ez)

    def load_field(
        self,
        namefile_Ex: str,
        namefile_Ey: str,
        namefile_Ez: str,
        namefile_potential: str,
        view: str | None = None,
    ):
        if view is None:
            path = self._drift_path()
        else:
            path = self._weighting_path(view)

        Ex = np.load(os.path.join(path, f"{namefile_Ex}.npy")).astype(np.float32)
        Ey = np.load(os.path.join(path, f"{namefile_Ey}.npy")).astype(np.float32)
        Ez = np.load(os.path.join(path, f"{namefile_Ez}.npy")).astype(np.float32)
        V = np.load(os.path.join(path, f"{namefile_potential}.npy")).astype(np.float32)

        return Ex, Ey, Ez, V
