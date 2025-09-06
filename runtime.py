from dataclasses import dataclass, field, asdict, astuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


@dataclass
class Op:
    dur: float = field(default=1.0)
    mem: float = field(default=0)
    color: str = field(default='#ffffff')
    text: str = field(default='')
    text_color: str = field(default='#ffffff')
    is_pad: bool = field(default=False)  # 是否为填充操作
    def __repr__(self):
        return f'{self.text}'

@dataclass
class P2P(Op):
    src: int = field(default=0)
    dst: int = field(default=0)
    def __repr__(self):
        return f'{self.src}->{self.dst}'
    
    def __eq__(self, value):
        return isinstance(value, P2P) and self.src == value.src and self.dst == value.dst and self.dur == value.dur


@dataclass
class Block:
    x: float
    y: float
    w: float
    h: float
    color: str
    text: str = field(default=None)
    text_color: str = field(default=None)
    op: Op = field(default=None)
    
@dataclass
class Device:
    id: int
    cur_ts: float = field(default=0)
    cur_mem: float = field(default=0)
    pending_list: list[Op] = field(default_factory=list)
    completed_list: list[Block] = field(default_factory=list)
    wait: Op = field(default=None)
    mem_history: list[float, float] = field(default_factory=lambda: [(0, 0)])  # (ts, mem)
    compute_total_time: float = field(default=0)
    
    def exec_op(self, op: Op):
        if not isinstance(op, P2P) and op.is_pad is False:
            self.compute_total_time += op.dur
        self.cur_ts += op.dur
        self.cur_mem += op.mem
        self.mem_history.append((self.cur_ts, self.cur_mem))
        return op
    
class Scheduler:
    def __init__(self, device_num: int):
        self.devices: list[Device] = [Device(i) for i in range(device_num)]

    def add_op(self, device_id: int, op: Op):
        self.devices[device_id].pending_list.append(op)
    
    def run(self, log=False, log_p2p=False):
        outputs = {device.id: '' for device in self.devices}
        while any([len(device.pending_list) > 0 or device.wait is not None for device in self.devices]):
            if all([len(device.pending_list) == 0 or device.wait is not None for device in self.devices]):
                print('Hang!')
                break
            for device in self.devices:
                while device.wait is None and device.pending_list:
                    op = device.pending_list.pop(0)
                    if isinstance(op, P2P):
                        device.wait = op
                        src_device = self.devices[op.src]
                        dst_device = self.devices[op.dst]
                        if src_device.wait is not None and src_device.wait == dst_device.wait:
                            start_ts = max(src_device.cur_ts, dst_device.cur_ts)
                            src_device.cur_ts = start_ts
                            dst_device.cur_ts = start_ts
                            src_device.completed_list.append(
                                Block(
                                    src_device.cur_ts, -src_device.id-1, src_device.wait.dur, 1, src_device.wait.color, None, None
                                )
                            )
                            dst_device.completed_list.append(
                                Block(
                                    dst_device.cur_ts, -dst_device.id-1, dst_device.wait.dur, 1, dst_device.wait.color, None, None
                                )
                            )
                            src_device.exec_op(src_device.wait)
                            dst_device.exec_op(dst_device.wait)
                            # region log
                            start_len = max(len(outputs[src_device.id]), len(outputs[dst_device.id]))
                            outputs[src_device.id] += ' ' * (start_len - len(outputs[src_device.id]))
                            outputs[dst_device.id] += ' ' * (start_len - len(outputs[dst_device.id]))
                            if log_p2p:
                                outputs[src_device.id] += f'{src_device.wait} '
                                outputs[dst_device.id] += f'{dst_device.wait} '
                            # endregion log
                            src_device.wait = None
                            dst_device.wait = None
                    elif isinstance(op, Op):
                        device.completed_list.append(
                            Block(
                                device.cur_ts, -device.id-1, op.dur, 1, op.color, op.text, op.text_color
                            )
                        )
                        device.exec_op(op)
                        # region log
                        outputs[device.id] += f'{op.text} '
                        # endregion log
            
        if log:
            print('\n'.join([f'Device {device_id}: {output}' for device_id, output in outputs.items()]))
            print(f'Schedule Bubble Rate: {1 - sum([device.compute_total_time for device in self.devices]) / max([device.cur_ts for device in self.devices]) / len(self.devices):.2%}, Max Allocated Mem: {", ".join([f"{max([mem for ts, mem in device.mem_history]):.2f}" for device in self.devices])}')

    def run_matplotlib(self, show_text: bool = True):
        self.run(log=True)
        fig, ax = plt.subplots(dpi=100)
        fig.canvas.manager.window.state('zoomed')
        for start_x, start_y, w, h, color, text, text_color, op in [astuple(block) for device in self.devices for block in device.completed_list]:
            if w <= 0 or h <= 0:
                continue
            ax.fill(
                [start_x, start_x + w, start_x + w, start_x],
                [start_y, start_y, start_y + h, start_y + h],
                color=color,
                edgecolor='black', linewidth=0.3,
            )
            if text is not None and show_text:
                ax.text(
                    start_x + w / 2, start_y + h / 2, text,
                    ha='center', va='center', fontsize=1/10*72, color=text_color
                )
        for device in self.devices:
            mem_history = device.mem_history
            x, y = zip(*mem_history)
            # x = np.repeat(x, 2)[1:].tolist() + [x[-1] + 1.0]
            # y = np.repeat(y, 2)[:-1].tolist() + [y[-1]]
            ax.plot(x, y, label=f'Device {device.id}')
            ax.fill_between(x, y, alpha=0.1)
        ax.set_title('Activation Memory Timeline', fontsize=16)
        ax.set_xlabel('Schedule Timeline', fontsize=16)
        yticks = ax.get_yticks()
        yticklabels = ax.get_yticklabels()
        filtered_yticks = yticks[yticks >= 0].tolist()
        filtered_yticklabels = [v for i, v in enumerate(yticklabels) if yticks[i] >= 0]
        for device in self.devices:
            filtered_yticks.append(-device.id - 0.5)
            filtered_yticklabels.append(f'Device {device.id}')
        ax.set_yticks(filtered_yticks)
        ax.set_yticklabels(filtered_yticklabels)
        ax.set_aspect(1)
        plt.legend(loc='upper left')
        plt.show()




if __name__ == '__main__':
    scheduler = Scheduler(4)
    scheduler.add_op(0, Op(dur=1, mem=1, color='#ff6b81', text='F1', text_color='#ffffff'))
    scheduler.add_op(0, P2P(src=0, dst=1, dur=0.1, color='#eccc68'))
    scheduler.add_op(0, P2P(src=3, dst=0, dur=0.1, color='#eccc68'))
    scheduler.add_op(0, Op(dur=1, mem=1, color='#ff6b81', text='F2', text_color='#ffffff'))

    scheduler.add_op(1, P2P(src=0, dst=1, dur=0.1, color='#eccc68'))
    scheduler.add_op(1, Op(dur=1, mem=1, color='#ff6b81', text='F1', text_color='#ffffff'))
    scheduler.add_op(1, P2P(src=1, dst=2, dur=0.1, color='#eccc68'))
    
    scheduler.add_op(2, P2P(src=1, dst=2, dur=0.1, color='#eccc68'))
    scheduler.add_op(2, Op(dur=1, mem=1, color='#ff6b81', text='F1', text_color='#ffffff'))
    scheduler.add_op(2, P2P(src=2, dst=3, dur=0.1, color='#eccc68'))
    
    scheduler.add_op(3, P2P(src=2, dst=3, dur=0.1, color='#eccc68'))
    scheduler.add_op(3, Op(dur=1, mem=1, color='#ff6b81', text='F1', text_color='#ffffff'))
    scheduler.add_op(3, P2P(src=3, dst=0, dur=0.1, color='#eccc68'))
    
    scheduler.run(log=True)
    scheduler.run_matplotlib()
