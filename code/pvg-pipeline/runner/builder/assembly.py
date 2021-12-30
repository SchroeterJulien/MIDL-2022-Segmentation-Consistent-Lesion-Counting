from typing import List, Callable, Tuple
from ignite.metrics import Metric
from ignite.engine import Events


class MetricAssembler():
	
    '''
    simple assembler based on assembly with custom attributes
    naming convension fix
    '''

    def __init__(self):

        # params
        self.metrics = []
        self.names = []
        
    def add(self, metric: Metric, name: str)->None:

        #safety 
        assert isinstance(metric, Metric), "Metric must be of type Metric"
        assert isinstance(name, str), "Name must be of type str"

        self.metrics.append(metric)
        self.names.append(name)

    def adds(self, metrics: List[Callable], names: List[str])->None:

        # safety
        assert isinstance(metrics, List), "Must add List of Metrics"
        assert isinstance(names, List), "Must add List of strs"
        assert len(metrics) == len(names), "Inputs must be of same length"

        for metric, name in zip(metrics, names):
            self.add(metric, name)

    def build(self)->Tuple:
        if len(self.metrics) == 0 or len(self.names) == 0: return None
        return (self.metrics, self.names)

class HandlerAssembler():
    '''
    simple assembler based on assembly with custom attributes
    naming convension fix
    '''

    def __init__(self):

        # params
        self.handlers = []
        self.states = []

    def add(self, handler: Callable, state: Events)->None:

        #safety 
        assert isinstance(handler, Callable), "Handler must be of type Handler"
        assert isinstance(state, Events), "State must be of type Events"

        self.handlers.append(handler)
        self.states.append(state)

    def adds(self, handlers: List[Callable], states: List[Events])->None:

        # safety
        assert isinstance(handlers, List), "Must add List of Callable handlers"
        assert isinstance(states, List), "Must add List of Events states"
        assert len(handlers) == len(states), "Inputs must be of same length"

        for handler, state in zip(handlers, states):
            self.add(handler, state)

    def build(self)->Tuple:
        if len(self.handlers) == 0 or len(self.states) == 0: return None
        return (self.handlers, self.states)
