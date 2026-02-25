from agents.aligen import AligenAgent
from agents.drift import DriftAgent
from agents.fql import FQLAgent
from agents.fql_bc import FQLAgent as FQLBCAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    aligen=AligenAgent,
    drift=DriftAgent,
    fql=FQLAgent,
    fql_bc=FQLBCAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
