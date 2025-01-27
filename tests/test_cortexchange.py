from cortexchange.wdclient import init_downloader
import os

# Make sure to store these variables.
login = os.getenv("WD_LOGIN")
password = os.getenv("WD_PASSWORD")


init_downloader(
    url="https://researchdrive.surfsara.nl/public.php/webdav/",
    cache="/home/larsve/.cache/cortexchange",
    login=login,
    password=password,
    # cache=".cache/cortexchange",
)

from cortexchange.architecture import get_architecture, Architecture

TransferLearning: type(Architecture) = get_architecture("surf/TransferLearning")
model = TransferLearning(device="cpu", model_name="surf/dinov2_09814")

print(model.args)

exit()

# torch_tensor = model.prepare_data(
#     "ILTJ160454.72+555949.7_selfcal/selfcal_007-MFS-image.fits"
# )
torch_tensor = model.prepare_data(
    "/scratch-shared/CORTEX/public.spider.surfsara.nl/lofarvwf/jdejong/CORTEX/calibrator_selection_robertjan/cnn_data/stop/ILTJ142906.77+334820.3_image_009-MFS-image.fits"
)
print(torch_tensor.shape)
result = model.predict(torch_tensor)
print(result)
