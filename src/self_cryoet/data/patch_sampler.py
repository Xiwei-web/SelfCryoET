from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


Index3D = Tuple[int, int, int]


@dataclass
class PatchSampler:
    volume_shape: Sequence[int]
    patch_size: Sequence[int]
    stride: Sequence[int]

    def generate(self) -> List[Index3D]:
        d, h, w = self.volume_shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride

        indices = []
        for z in self._positions(d, pd, sd):
            for y in self._positions(h, ph, sh):
                for x in self._positions(w, pw, sw):
                    indices.append((z, y, x))
        return indices

    @staticmethod
    def _positions(size: int, patch: int, stride: int) -> Iterable[int]:
        if patch >= size:
            return [0]
        pos = list(range(0, size - patch + 1, stride))
        if pos[-1] != size - patch:
            pos.append(size - patch)
        return pos

