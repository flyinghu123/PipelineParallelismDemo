from dataclasses import dataclass, field, astuple
from runtime import Scheduler, Op, P2P


pp_size = 4
num_microbatches = 4


scheduler = Scheduler(pp_size)

@dataclass(kw_only=True)
class ForwardOp(Op):
    dur: float = 1.0
    mem: float = 1.0
    color: str = '#4285F4'

@dataclass(kw_only=True)
class BackwardOp(Op):
    dur: float = 2.0
    mem: float = -1.0
    color: str = '#E69138'

@dataclass(kw_only=True)
class SendRecvOp(P2P):
    dur: float = 0.
    mem: float = 0.0
    color: str = '#cde4dd'

for stage_id in range(pp_size):
    for i in range(num_microbatches):
        if stage_id != 0:
            # recv forward
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id-1, dst=stage_id))
        # forward
        scheduler.add_op(stage_id, ForwardOp(text=f'F{i}'))
        if stage_id != pp_size - 1:
            # send forward
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=stage_id+1))
    for i in range(num_microbatches):
        if stage_id != pp_size - 1:
            # recv backward
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id+1, dst=stage_id))
        # backward
        scheduler.add_op(stage_id, BackwardOp(text=f'B{i}'))
        if stage_id != 0:
            # send backward
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=stage_id-1))
scheduler.run_matplotlib()