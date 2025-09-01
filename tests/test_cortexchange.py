# import socket
# import os

# # Optional opt-out: set ALLOW_NETWORK=1 to disable this block.
# if os.getenv("ALLOW_NETWORK") == "1":
#     raise SystemExit  # do nothing; exit this file cleanly

# class _BlockedSocket(socket.socket):
#     def __init__(self, *args, **kwargs):
#         raise OSError("Network disabled by sitecustomize.py")

# def _deny(*args, **kwargs):
#     raise OSError("Network disabled by sitecustomize.py")

# # Block new TCP/UDP sockets
# socket.socket = _BlockedSocket  # type: ignore

# # Block name resolution (prevents DNS attempts even if someone restores socket)
# socket.getaddrinfo = _deny      # type: ignore
# socket.gethostbyname = _deny    # type: ignore
# socket.gethostbyname_ex = _deny # type: ignore
# socket.create_connection = _deny # type: ignore

# # Also block urllib / http.client fast paths (helps when libraries bypass socket.create_connection)
# try:
#     import http.client as _hc
#     _hc.HTTPConnection.connect = _deny   # type: ignore
#     _hc.HTTPSConnection.connect = _deny  # type: ignore
# except Exception:
#     pass

from cortexchange.wdclient import init_downloader
import os
import warnings


# Suppress specific xFormers-related warnings
warnings.filterwarnings(
    "ignore",
    message="xFormers is available*",
    category=UserWarning,
)

# Make sure to store these variables.
login = os.getenv("WD_LOGIN")
password = os.getenv("WD_PASSWORD")



init_downloader(
    url="https://researchdrive.surfsara.nl/public.php/webdav/",
    cache=f"{os.path.join(os.path.expanduser('~'), '.cache/cortexchange')}",
    login=login,
    password=password,
)
from cortexchange.architecture import get_architecture, Architecture

TransferLearning: type(Architecture) = get_architecture("surf/TransferLearningV2")
model = TransferLearning(
    device="cuda", model_name="surf/dino_big_lora_tune_posclsreg_may_O2_aug_099"
)


# torch_tensor = model.prepare_data(
#     "ILTJ160454.72+555949.7_selfcal/selfcal_007-MFS-image.fits"
# )
# for i in range(56, 2044, 56):
#     torch_tensor = model.prepare_data(
#         "/scratch-shared/CORTEX/public.spider.surfsara.nl/lofarvwf/jdejong/CORTEX/calibrator_selection_robertjan/cnn_data/stop/ILTJ142906.77+334820.3_image_009-MFS-image.fits",
#         resize=i,
#     )
#     print(torch_tensor.shape)
#     pred, _ = model.predict(torch_tensor)
#     print(pred.item())
