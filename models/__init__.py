from .Attn import Attn
from .BiLSTM import BiLSTM
# from .CNNText import CNNText
from .EmbedText import EmbedText
from .MatchingNetwork import MatchingNetwork
from .PairCosineSim import PairCosineSim
from .Weight_Init import weight_init

# These are not required
del Attn
del BiLSTM
# del CNNText
del EmbedText
del MatchingNetwork
del PairCosineSim
