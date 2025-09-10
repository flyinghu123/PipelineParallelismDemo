# 该代码删除大量注释, 实现参考https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py

from dataclasses import dataclass, field, astuple
from runtime import Scheduler, Op, P2P
import matplotlib.colors as mcolors
import numpy as np

pp_size = 4
vpp_size = 2
num_microbatches = 8
microbatch_group_size_per_vp_stage = 4

assert num_microbatches % microbatch_group_size_per_vp_stage == 0 or num_microbatches % microbatch_group_size_per_vp_stage >= pp_size
assert microbatch_group_size_per_vp_stage <= num_microbatches and microbatch_group_size_per_vp_stage >= pp_size

scheduler = Scheduler(pp_size)

# forward chunk colors
f_colors = [mcolors.to_hex(np.array(mcolors.to_rgb('#305496')) * (1 - i) + np.array(mcolors.to_rgb('#B4C6E7')) * i)
            for i in np.linspace(0, 1, vpp_size)]
# backward chunk colors
b_colors = [mcolors.to_hex(np.array(mcolors.to_rgb('#375623')) * (1 - i) + np.array(mcolors.to_rgb('#A9D08E')) * i)
            for i in np.linspace(0, 1, vpp_size)]

@dataclass(kw_only=True)
class ForwardOp(Op):
    dur: float = 1.0
    mem: float = 1.0 / vpp_size

@dataclass(kw_only=True)
class BackwardOp(Op):
    dur: float = 2.0
    mem: float = -1.0 / vpp_size

@dataclass(kw_only=True)
class SendRecvOp(P2P):
    dur: float = 0.
    mem: float = 0.0
    color: str = '#EDEDED'
    

# 奇偶通信策略
def _p2p_ops(send_prev: bool = False, recv_prev: bool = False, send_next: bool = False, recv_next: bool = False):
    if stage_id % 2 == 0:
        if send_next:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=(stage_id+1) % pp_size))
        if recv_prev:
            scheduler.add_op(stage_id, SendRecvOp(src=(stage_id-1) % pp_size, dst=stage_id))
        if send_prev:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=(stage_id-1) % pp_size))
        if recv_next:
            scheduler.add_op(stage_id, SendRecvOp(src=(stage_id+1) % pp_size, dst=stage_id))
    else:
        if recv_prev:
            scheduler.add_op(stage_id, SendRecvOp(src=(stage_id-1) % pp_size, dst=stage_id))
        if send_next:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=(stage_id+1) % pp_size))
        if recv_next:
            scheduler.add_op(stage_id, SendRecvOp(src=(stage_id+1) % pp_size, dst=stage_id))
        if send_prev:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id, dst=(stage_id-1) % pp_size))


def send_forward_recv_forward(send_next: bool = False, recv_prev: bool = False):
    _p2p_ops(send_next=send_next, recv_prev=recv_prev)

def send_backward_recv_backward(send_prev: bool = False, recv_next: bool = False):
    _p2p_ops(send_prev=send_prev, recv_next=recv_next)
    
def send_forward_backward_recv_forward_backward(send_next: bool = False, send_prev: bool = False, recv_prev: bool = False, recv_next: bool = False):
    _p2p_ops(send_next=send_next, send_prev=send_prev, recv_prev=recv_prev, recv_next=recv_next)

def get_model_chunk_id(virtual_microbatch_id, forward):
    """Helper method to get the model chunk ID given the iteration number."""
    model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
    if not forward:
        model_chunk_id = vpp_size - model_chunk_id - 1
    return model_chunk_id

def get_microbatch_id_in_model_chunk(iteration_id, forward):
    """Helper method to get the microbatch_id within model chunk given the iteration number."""
    microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
    return microbatch_id_in_model_chunk

def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
    # 中间的stage都需要recv前面stage的forward
    # forward情况下stage 0, 需要等待last stage开始forward才开始接受
    # 如下pp_size = 4情况下, stage 0需在3后recv, 考虑具体实现是在forward判断是否需要recv, 故在3的判断返回True
    # 同时当last stage的chunk_id为最后一个时, stage 0不需要recv
    # 0 1 2 3 -> recv
    #   1 2 3
    #     2 3
    #       3 -> send
    # backward同理
    recv = True
    is_leading_pipeline_stage = (
        stage_id == 0
        if forward
        else stage_id == pp_size - 1
    )
    last_model_chunk = (vpp_size - 1) if forward else 0
    if is_leading_pipeline_stage:
        if virtual_microbatch_id < (pp_size - 1):
            recv = False
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
        else:
            next_model_chunk_id = get_model_chunk_id(
                virtual_microbatch_id - (pp_size - 1), forward
            )
        if next_model_chunk_id == last_model_chunk:
            recv = False
        if forward:
            next_model_chunk_id += 1
        else:
            next_model_chunk_id -= 1
    else:
        next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

    return recv

for stage_id in range(pp_size):
    # 计算warmup阶段的microbatch数量, (vpp_size - 1)组, last stage进入1f1b,
    # 其他stage多forward和backward串行依各pp_size - stage_id - 1
    total_num_microbatches = num_microbatches * vpp_size
    num_warmup_microbatches = (pp_size - stage_id - 1) * 2
    num_warmup_microbatches += (vpp_size - 1) * microbatch_group_size_per_vp_stage
    are_all_microbatches_in_warmup = num_warmup_microbatches >= total_num_microbatches
    num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
    
    # 计算virtual microbatch to real microbatch id
    microbatch_id_table = [i+k for i in range(0, num_microbatches, microbatch_group_size_per_vp_stage) for j in range(vpp_size) for k in range(min(microbatch_group_size_per_vp_stage, num_microbatches - i))]
    # 计算virtual microbatch to chunk id
    model_chunk_id_table = [j for i in range(0, num_microbatches, microbatch_group_size_per_vp_stage) for j in range(vpp_size) for k in range(min(microbatch_group_size_per_vp_stage, num_microbatches - i))]

    # warmup
    # recv forward
    if stage_id != 0:
        scheduler.add_op(stage_id, SendRecvOp(src=stage_id-1, dst=stage_id))
    for k in range(num_warmup_microbatches):
        # 根据分组交替执行规则, 计算k对应的chunk_id
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        # 根据分组交替执行规则, 计算k对应的microbatch_id
        microbatch_id = get_microbatch_id_in_model_chunk(k, forward=True)
        recv_prev = recv_tensor_from_previous_stage(k, forward=True)
        if k == total_num_microbatches - 1:
            recv_prev = False
        
        # forward
        scheduler.add_op(stage_id, ForwardOp(text=f'F{microbatch_id}', color=f_colors[cur_model_chunk_id]))
        
        send_next = not (stage_id == pp_size - 1 and cur_model_chunk_id == vpp_size - 1)
        
        if k == num_warmup_microbatches - 1 and not are_all_microbatches_in_warmup:
            # send forward, recv forward, recv backward
            send_forward_backward_recv_forward_backward(send_next=send_next,
                                                        send_prev=False,
                                                        recv_prev=recv_prev,
                                                        recv_next=stage_id != pp_size - 1)
        else:
            # send forward, recv backward
            send_forward_recv_forward(send_next=send_next, recv_prev=recv_prev)
            
    # 1f1b
    for k in range(num_microbatches_remaining):
        forward_k = k + num_warmup_microbatches
        cur_forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        forward_microbatch_id = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        backward_k = k
        cur_backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        backward_microbatch_id = get_microbatch_id_in_model_chunk(backward_k, forward=False)
        # forward
        scheduler.add_op(stage_id, ForwardOp(text=f'F{forward_microbatch_id}', color=f_colors[cur_forward_model_chunk_id]))
        # backward
        scheduler.add_op(stage_id, BackwardOp(text=f'B{backward_microbatch_id}', color=b_colors[cur_backward_model_chunk_id]))

        send_next = not (stage_id == pp_size - 1 and cur_forward_model_chunk_id == vpp_size - 1)
        send_prev = not (stage_id == 0 and cur_backward_model_chunk_id == 0)

        recv_prev = recv_tensor_from_previous_stage(forward_k, forward=True)
        recv_next = recv_tensor_from_previous_stage(backward_k, forward=False)
        if k == num_microbatches_remaining - 1:
            recv_prev = False
        send_forward_backward_recv_forward_backward(
            send_next=send_next,
            send_prev=send_prev,
            recv_prev=recv_prev,
            recv_next=recv_next
        )
    # cooldown
    if are_all_microbatches_in_warmup:
        # recv backward
        if stage_id != pp_size - 1:
            scheduler.add_op(stage_id, SendRecvOp(src=stage_id+1, dst=stage_id))
    for k in range(num_microbatches_remaining, total_num_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=False)
        microbatch_id = get_microbatch_id_in_model_chunk(k, forward=False)
        recv_next = recv_tensor_from_previous_stage(k, forward=False)
        if k == (total_num_microbatches - 1):
            recv_next = False
        # backward
        scheduler.add_op(stage_id, BackwardOp(text=f'B{microbatch_id}', color=b_colors[cur_model_chunk_id]))
        
        send_prev = not (stage_id == 0 and cur_model_chunk_id == 0)
        
        # send backward, recv backward
        send_backward_recv_backward(send_prev=send_prev, recv_next=recv_next)
scheduler.run_matplotlib(legends=[(f'F-Chunk{i}', f_colors[i]) for i in range(vpp_size)] + \
                         [(f'B-Chunk{i}', b_colors[i]) for i in range(vpp_size)] + \
                         [('SendRecv', SendRecvOp.color)])
