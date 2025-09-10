from dataclasses import dataclass, field, astuple
from runtime import Scheduler, Op, P2P


pp_size = 4
num_microbatches = 8


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
    color: str = '#EDEDED'
    

# 奇偶通信策略
def _p2p_ops(send_prev: bool = False, recv_prev: bool = False, send_next: bool = False, recv_next: bool = False):
    if stage_id % 2 == 0:
        if send_next:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=stage_id+1))
        if recv_prev:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id-1, dst=stage_id))
        if send_prev:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=stage_id-1))
        if recv_next:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id+1, dst=stage_id))
    else:
        if recv_prev:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id-1, dst=stage_id))
        if send_next:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=stage_id+1))
        if recv_next:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id+1, dst=stage_id))
        if send_prev:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=stage_id-1))


def send_forward_recv_backward():
    if stage_id == pp_size - 1:
        return
    _p2p_ops(send_next=True, recv_next=True)

def send_backward_recv_forward():
    if stage_id == 0:
        return
    _p2p_ops(send_prev=True, recv_prev=True)
    
    
for stage_id in range(pp_size):
    num_warmup_microbatches = pp_size - stage_id - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches
    # warmup
    for i in range(num_warmup_microbatches):
        if stage_id != 0:
            # recv forward
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id-1, dst=stage_id))
        # forward
        scheduler.add_op(stage_id, ForwardOp(text=f'F{i}'))
        if stage_id != pp_size - 1:
            # send forward
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=stage_id+1))
    if num_microbatches_remaining > 0 and stage_id != 0:
        # recv forward
        scheduler.add_op(stage_id, SendRecvOp(src=stage_id-1, dst=stage_id))
    # 推迟forward, 使得图像1f1b的第一个F在B前一刻
    # scheduler.add_op(stage_id, Op(dur=BackwardOp.dur * (pp_size-1-stage_id), is_pad=True))
    # 1f1b
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)
        # forward
        scheduler.add_op(stage_id, ForwardOp(text=f'F{i+num_warmup_microbatches}'))
        # send forward + recv backward
        send_forward_recv_backward()
        # backward
        scheduler.add_op(stage_id, BackwardOp(text=f'B{i}'))
        if last_iteration:
            # send backward
            if stage_id != 0:
                scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=stage_id-1))
        else:
            # send backward + recv forward
            send_backward_recv_forward()
        
    # cooldown:
    for i in range(num_warmup_microbatches):
        if stage_id != pp_size - 1:
            # recv backward
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id+1, dst=stage_id))
        # backward
        scheduler.add_op(stage_id, BackwardOp(text=f'B{i+num_microbatches_remaining}'))
        if stage_id != 0:
            # send backward
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=stage_id-1))
scheduler.run_matplotlib(legends=[('Forward', ForwardOp.color), ('Backward', BackwardOp.color), ('SendRecv', SendRecvOp.color)])