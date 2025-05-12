from astronnomy.pre_processing_for_ml import *
from astronnomy.training.train_nn import MultiEpochsDataLoader, seed_worker
from typing import Tuple


class FitsAndSourceDataset(FitsDataset):
    def __init__(self, root_dir, mode="train"):
        super().__init__(root_dir, mode)

        self.sources = [
            source if source != "source" else self.data_paths[i].split("/")[-1]
            for i, source in enumerate(
                [
                    npy_path.split("/")[-1].split("_0")[0].split("-0")[0]
                    for npy_path in self.data_paths
                ]
            )
        ]

        self.unique_sources, self.source_inds, self.source_counts = np.unique(
            self.sources, return_inverse=True, return_counts=True
        )

        self.cal_step = np.asarray(
            [
                int(step) if step.isdigit() else -1
                for step in (
                    npy_path.split("/")[-1].split("_0")[-1].split("-0")[-1][:2]
                    for npy_path in self.data_paths
                )
            ]
        )

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, int, int]:
        npy_path = self.data_paths[idx]
        # print(npy_path.split("/")[-1].split("_0")[-1].split("-0")[-1][:2])
        cal_step = self.cal_step[idx]
        source_ind = self.source_inds[idx]
        label = self.labels[idx]
        image_data = np.load(npy_path)["arr_0"]  # there is always only one array

        # Pre-processing
        image_data = self.transform_data(image_data)

        return image_data, label, source_ind, cal_step


def get_dataloaders_source(dataset_root, batch_size):
    num_workers = min(18, len(os.sched_getaffinity(0)))

    prefetch_factor, persistent_workers = (
        (2, True) if num_workers > 0 else (None, False)
    )
    generators = {}
    for mode in ("val", "train"):
        generators[mode] = torch.Generator()

    loaders = tuple(
        MultiEpochsDataLoader(
            dataset=FitsAndSourceDataset(dataset_root, mode=mode),
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            worker_init_fn=seed_worker,
            generator=generators[mode],
            pin_memory=True,
            shuffle=True if mode == "train" else False,
            drop_last=True if mode == "train" else False,
        )
        for mode in ("train", "val")
    )

    return loaders


if __name__ == "__main__":
    get_dataloaders_source(
        f"/scratch-shared/CORTEX/public.spider.surfsara.nl/lofarvwf/jdejong/CORTEX/calibrator_selection_robertjan/cnn_data",
        64,
    )

    # for img, label, station, cal_step in loader:
    #     print(station, cal_step)
