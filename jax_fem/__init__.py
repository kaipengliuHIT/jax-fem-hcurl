from pyfiglet import Figlet

f = Figlet(font='starwars')
print(f.renderText('JAX - FEM'))

from .logger_setup import setup_logger
# LOGGING
logger = setup_logger(__name__)

# H(curl) elements support
from .basis import is_hcurl_element
from .hcurl_fe import HCurlFiniteElement, compute_curl_curl_matrix, compute_mass_matrix

# TODO: Be automatic
# __version__ = "0.0.11"