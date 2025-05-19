import enum


class Methods(enum.StrEnum):
    Mipn = "MiPN"
    Mipnc = "MiPN(c)"
    Venas = "VeNAS"
    Venasc = "VeNAS(c)"
    Rlboa = "RLBOA"
    Rlboac = "RLBOA(c)"
    TimeUtility = "TimeUtility"
    Sengupta = "Sengupta"
    Senguptad = "Sengupta(d)"


class BaselineMethods(enum.StrEnum):
    Boulware = "Boulware"
    Conceder = "Conceder"
    Linear = "Linear"
    Hardheaded = "Hardheaded"
    Atlas3 = "Atlas3"
    AgentK = "AgentK"
    CUHK = "CUHK"
    AgentGG = "AgentGG"
    NiceTitForTat = "NiceTitForTat"
    NaiveTitForTat = "NaiveTitForTat"
    AverageTitForTat = "AverageTitForTat"
