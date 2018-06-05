import config

from .generic import GenericModel

# classname of NNRF-ST: NNRF
from .nnrf import NNRF

# classname of NNRF-MT: MTRF
from .nnrf_mt import MTRF

# classname of RoFA-MT: MTRFv4
from .rofa import MTRFv4

# classname of ResRoFA-MT: MTRFv4Res
from .resrofa import MTRFv4Res

# ablation study: sigle task version for composition methods

# classname of RoFA-ST: NNRF_ROFA
from .rofa_st import NNRF_ROFA

# classname of ResRoFA-ST: NNRF_ResROFA
from .resrofa_st import NNRF_ResROFA