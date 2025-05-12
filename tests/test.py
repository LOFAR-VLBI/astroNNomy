from astronnomy.training.train_nn import get_dataloaders

loaders = get_dataloaders(
    "/scratch-shared/CORTEX/public.spider.surfsara.nl/lofarvwf/jdejong/CORTEX/calibrator_selection_robertjan/cnn_data/",
    12,
)
